import os
import gradio as gr
import fitz
import tempfile
from PIL import Image
import io

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class PDFChatbot:
    def __init__(self, api_key=None):
        self.current_pdf = None
        self.pdf_text = ""
        self.page_texts = []
        self.current_page = 0
        self.total_pages = 0
        self.api_key = api_key

        template = """다음 텍스트를 한국어로 번역해주세요. 
        번역할 텍스트는 다음과 같습니다:
        {text}
        
        번역할 때 다음 사항을 지켜주세요:
        1. 전문적이고 자연스러운 한국어로 번역
        2. 원문의 의미를 정확하게 전달
        3. 전문 용어가 있다면 적절한 한국어 용어 사용
        4. 문맥을 고려한 자연스러운 번역
        """

        self.translate_prompt = PromptTemplate(
            template=template, input_variables=["text"]
        )

        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    def set_api_key(self, api_key):
        """Function to set or update the API key"""
        if not api_key:
            return "Please enter an API key"
        try:
            self.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            return "API Key has been updated successfully!"
        except Exception as e:
            return f"Error setting API key: {str(e)}"

    def load_pdf(self, file_obj):
        """Function to load PDF and extract text"""
        if file_obj is None:
            return None, None, "Please upload a PDF file."

        try:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "temp.pdf")

            with open(file_obj.name, "rb") as f:
                pdf_content = f.read()

            with open(temp_path, "wb") as f:
                f.write(pdf_content)

            self.current_pdf = fitz.open(temp_path)
            self.page_texts = []
            self.total_pages = len(self.current_pdf)
            self.current_page = 0

            # Extract text from each page
            for page in self.current_pdf:
                self.page_texts.append(page.get_text())

            self.pdf_text = "\n".join(self.page_texts)

            # Generate first page image
            first_page_image = self.get_page_image(0)

            os.remove(temp_path)
            os.rmdir(temp_dir)

            return (
                first_page_image,
                None,
                f"PDF loaded successfully. Total {self.total_pages} pages.",
            )
        except Exception as e:
            return None, None, f"Error loading PDF: {str(e)}"

    def get_page_image(self, page_num):
        """Function to generate image for a specific page"""
        if not self.current_pdf or page_num < 0 or page_num >= self.total_pages:
            return None

        try:
            page = self.current_pdf[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x magnification
            img_data = pix.tobytes("png")

            # Convert to PIL Image object
            img = Image.open(io.BytesIO(img_data))
            return img

        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def change_page(self, page_num):
        """Function to change page"""
        if not self.current_pdf or page_num < 0 or page_num >= self.total_pages:
            return None

        self.current_page = page_num
        return self.get_page_image(page_num)

    def translate_page(self, page_number):
        """Function to translate a specific page using GPT-4"""
        if not self.api_key:
            return "Please set your OpenAI API key first."

        if (
            not self.current_pdf
            or page_number < 0
            or page_number >= len(self.page_texts)
        ):
            return "Invalid page number."

        try:
            page_text = self.page_texts[int(page_number)]
            prompt = self.translate_prompt.format(text=page_text)
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            return f"Error during translation: {str(e)}"


def create_interface():
    pdf_chatbot = PDFChatbot()

    with gr.Blocks() as demo:
        gr.Markdown("# PDF Translator")

        with gr.Row():
            # API Key input at the top
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key here",
                show_label=True,
            )

        with gr.Row():
            # Left Column: PDF Upload and Preview
            with gr.Column(scale=1):
                # Compress top controls into one line
                with gr.Row():
                    with gr.Column(scale=2):
                        pdf_file = gr.File(
                            label="Upload PDF File",
                            file_types=[".pdf"],
                            elem_classes="compact-file-upload",
                        )
                    with gr.Column(scale=1):
                        upload_status = gr.Textbox(
                            label="Status", show_label=False, container=False
                        )

                # PDF Viewer
                pdf_viewer = gr.Image(
                    label="PDF Viewer", show_label=False, height=750, container=False
                )

                # Compact bottom navigation
                with gr.Row():
                    prev_btn = gr.Button("◀", size="sm")
                    current_page = gr.Number(
                        value=0,
                        minimum=0,
                        step=1,
                        container=False,
                        show_label=False,
                        elem_classes="page-number-input",
                    )
                    next_btn = gr.Button("▶", size="sm")

            # Right Column: Translation
            with gr.Column(scale=1):
                with gr.Row():
                    translate_btn = gr.Button("Translate", size="sm")

                translated_text = gr.Textbox(
                    label="Translation Result",
                    lines=30,
                    show_label=False,
                    container=False,
                )

        gr.Markdown(
            """
        <style>
        .compact-file-upload {
            padding: 0;
            margin: 0;
        }
        .page-number-input {
            width: 60px !important;
            padding: 0 8px !important;
        }
        </style>
        """
        )

        # Event Handler for API Key
        def update_api_key(api_key):
            return pdf_chatbot.set_api_key(api_key)

        api_key_input.change(
            fn=update_api_key, inputs=[api_key_input], outputs=[upload_status]
        )

        # Event handlers for PDF operations
        def on_pdf_load(file_obj):
            if not pdf_chatbot.api_key:
                return None, None, "Please set your OpenAI API key first."
            result = pdf_chatbot.load_pdf(file_obj)
            pdf_chatbot.current_page = 0
            return result

        pdf_file.change(
            fn=on_pdf_load,
            inputs=[pdf_file],
            outputs=[pdf_viewer, translated_text, upload_status],
        )

        def prev_page():
            pdf_chatbot.current_page = max(0, pdf_chatbot.current_page - 1)
            image = pdf_chatbot.change_page(pdf_chatbot.current_page)
            return image, pdf_chatbot.current_page

        def next_page():
            pdf_chatbot.current_page = min(
                pdf_chatbot.total_pages - 1, pdf_chatbot.current_page + 1
            )
            image = pdf_chatbot.change_page(pdf_chatbot.current_page)
            return image, pdf_chatbot.current_page

        prev_btn.click(fn=prev_page, outputs=[pdf_viewer, current_page])

        next_btn.click(fn=next_page, outputs=[pdf_viewer, current_page])

        def on_page_number_change(page_num):
            if page_num is None:
                page_num = 0
            page_num = int(page_num)
            if pdf_chatbot.current_pdf:
                page_num = max(0, min(page_num, pdf_chatbot.total_pages - 1))
                pdf_chatbot.current_page = page_num
                return pdf_chatbot.change_page(page_num)
            return None

        current_page.change(
            fn=on_page_number_change, inputs=[current_page], outputs=[pdf_viewer]
        )

        def on_translate():
            if not pdf_chatbot.api_key:
                return "Please set your OpenAI API key first."
            return pdf_chatbot.translate_page(pdf_chatbot.current_page)

        translate_btn.click(fn=on_translate, outputs=[translated_text])

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
