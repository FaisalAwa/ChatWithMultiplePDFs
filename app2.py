# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS  # for vector embeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitters = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitters.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks,embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "Answer is not available in the context." Do not provide any information beyond the context.
#     Context: \n{context}\n
#     Question: \n{question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings = embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     st.write("Reply: ", response["output_text"])

# def main():
#     st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")
#     st.header("Chat with Multiple PDFs using Gemini")

#     user_question = st.text_input("Ask a question from the PDF files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu")
#         pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS  # for vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitters.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "Answer is not available in the context." Do not provide any information beyond the context.
    Context: \n{context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def add_custom_css():
    st.markdown(
        """
        <style>
        .header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #FFFF00;
        }
        .subheader {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #FF0000;
        }
        .description {
            font-size: 18px;
            color: #AAAAAA;
        }
        body {
            background-color: #001f3f;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

def main():
    # st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")
    st.markdown("""
    <svg width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
      <g>
        <title>Document Q & A</title>
        <rect width="100%" height="200" fill="#001f3f" />
        <circle cx="150" cy="100" r="80" fill="#FF0000" />
        <text x="150" y="115" font-size="35" font-family="Arial" fill="#FFFFFF" text-anchor="middle">📚</text>
        <text x="50%" y="180" font-size="24" font-family="Arial" fill="#FFFF00" text-anchor="middle">
          <animate attributeName="opacity" values="0;1;0" dur="3s" repeatCount="indefinite" />
          Chat with Multiple PDFs using Gemini
        </text>
      </g>
    </svg>
    """, unsafe_allow_html=True)
    st.markdown('<div class="header">Chat with Multiple PDFs using Gemini 📚🤖</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Retrieve and Analyze Information from Your Documents</div>', unsafe_allow_html=True)

    st.write("""
    ### Project Description:
    This application uses the power of the Gemini model to retrieve and analyze information from your documents. Simply ask a question, and the model will search the most relevant information from the documents in the backend. Additionally, it performs a similarity search across all documents to provide the best answers.
    """)

    user_question = st.text_input("Ask a question from the PDF files 📄")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu 📂")
        pdf_docs = st.file_uploader("Upload your PDF files 📑", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing... ⏳"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done ✅")

if __name__ == "__main__":
    main()
