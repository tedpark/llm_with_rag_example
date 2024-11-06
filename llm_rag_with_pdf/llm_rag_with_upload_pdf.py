import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader


class PDFChatbot:
    def __init__(self):
        self.qa_chain = None
        self.api_key = None

    def initialize_qa_chain(self, pdf_path):
        if not self.api_key:
            return "Please enter your OpenAI API key first!"

        # Load PDF and split text
        loader = PyPDFLoader(pdf_path)
        texts = loader.load_and_split()

        # Setup embeddings and vector DB
        embedding = OpenAIEmbeddings(openai_api_key=self.api_key)
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
        )

        # Setup retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})

        # Setup prompt template
        template = """
        If the content is not in the search results, please let me know that you cannot answer. Please respond in a friendly, casual tone.
        {context}

        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        # Setup LLM
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=self.api_key)

        # Create QA chain
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs={"prompt": prompt},
            retriever=retriever,
            return_source_documents=True,
        )

    def set_api_key(self, api_key):
        if not api_key:
            return "Please enter your OpenAI API key."
        try:
            self.api_key = api_key
            return "API key set successfully!"
        except Exception as e:
            return f"Error setting API key: {str(e)}"

    def load_pdf(self, file):
        if not self.api_key:
            return "Please enter your OpenAI API key first!"
        if file is None:
            return "Please upload a PDF file."
        try:
            self.qa_chain = self.initialize_qa_chain(file.name)
            return "PDF file loaded successfully!"
        except Exception as e:
            return f"Error loading PDF: {str(e)}"

    def get_response(self, message):
        if not self.api_key:
            return "Please enter your OpenAI API key first!"
        if self.qa_chain is None:
            return "Please upload a PDF file first!"
        try:
            response = self.qa_chain.invoke(message)
            return response["result"].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


# Create interface
with gr.Blocks() as demo:
    chatbot_instance = PDFChatbot()

    with gr.Row():
        # API Key input
        api_key_input = gr.Textbox(
            label="Enter OpenAI API Key", type="password", placeholder="sk-..."
        )
        api_status = gr.Textbox(label="API Key Status")

    with gr.Row():
        # PDF upload section
        with gr.Column(scale=1):
            pdf_file = gr.File(label="Upload PDF File", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Upload Status")

        # Chat section
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="PDF-based Chatbot")
            msg = gr.Textbox(label="Ask a question!")
            with gr.Row():
                submit = gr.Button("Send")
                clear = gr.Button("Clear Chat")

    # Handle API key setting
    def handle_api_key(api_key):
        return chatbot_instance.set_api_key(api_key)

    # Handle PDF upload
    def handle_file_upload(file):
        return chatbot_instance.load_pdf(file)

    # Handle chat responses
    def respond(message, chat_history):
        bot_message = chatbot_instance.get_response(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    # Connect events
    api_key_input.change(
        fn=handle_api_key, inputs=[api_key_input], outputs=[api_status]
    )

    pdf_file.upload(fn=handle_file_upload, inputs=[pdf_file], outputs=[upload_status])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch interface
demo.launch(debug=True)
