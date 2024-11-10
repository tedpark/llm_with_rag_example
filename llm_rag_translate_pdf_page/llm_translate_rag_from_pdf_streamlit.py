import os
import streamlit as st
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
            self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)

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
            return None, "Please upload a PDF file."

        try:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "temp.pdf")

            bytes_data = file_obj.read()
            with open(temp_path, "wb") as f:
                f.write(bytes_data)

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
                f"PDF loaded successfully. Total {self.total_pages} pages.",
            )
        except Exception as e:
            return None, f"Error loading PDF: {str(e)}"

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


def main():
    st.set_page_config(layout="wide")
    st.title("PDF Translator")

    # Initialize session state
    if "pdf_chatbot" not in st.session_state:
        st.session_state.pdf_chatbot = PDFChatbot()

    if "current_page" not in st.session_state:
        st.session_state.current_page = 0

    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        status = st.session_state.pdf_chatbot.set_api_key(api_key)
        st.info(status)

    # Create two columns for the layout
    col1, col2 = st.columns(2)

    with col1:
        # PDF Upload
        uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])

        if uploaded_file:
            image, status = st.session_state.pdf_chatbot.load_pdf(uploaded_file)
            st.info(status)

            if image:
                # Page navigation
                col_prev, col_page, col_next = st.columns([1, 2, 1])

                with col_prev:
                    if st.button("◀"):
                        st.session_state.current_page = max(
                            0, st.session_state.current_page - 1
                        )

                with col_page:
                    page_num = st.number_input(
                        "Page",
                        min_value=0,
                        max_value=st.session_state.pdf_chatbot.total_pages - 1,
                        value=st.session_state.current_page,
                        step=1,
                    )
                    if page_num != st.session_state.current_page:
                        st.session_state.current_page = page_num

                with col_next:
                    if st.button("▶"):
                        st.session_state.current_page = min(
                            st.session_state.pdf_chatbot.total_pages - 1,
                            st.session_state.current_page + 1,
                        )

                # Display PDF page
                current_image = st.session_state.pdf_chatbot.get_page_image(
                    st.session_state.current_page
                )
                if current_image:
                    st.image(current_image, use_column_width=True)

    with col2:
        if uploaded_file:
            if st.button("Translate Current Page"):
                translation = st.session_state.pdf_chatbot.translate_page(
                    st.session_state.current_page
                )
                st.text_area("Translation Result", translation, height=600)


if __name__ == "__main__":
    main()
