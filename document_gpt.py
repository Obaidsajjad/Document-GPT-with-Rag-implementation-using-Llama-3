import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain_community.callbacks.manager import get_openai_callback
from langchain.prompts import PromptTemplate
# from qdrant_client import QdrantClient

groq_api_key="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"

def main():
    load_dotenv('.env')
    st.set_page_config("Document GPT")
    st.header("Lets Chat with your Documents")
    st.subheader("Ask Qusetions about your Documents ... ")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processcomplete" not in st.session_state:
        st.session_state.processcomplete = None


    with st.sidebar:
        uploaded_doc=st.file_uploader("Upload your files ... ", type=['pdf','docx'],accept_multiple_files=True)
        process=st.button("Process")

    if process:
        if not groq_api_key:
            st.error("Something went wrong with API key")
            st.stop()
        if not uploaded_doc:
            st.warning("Upload Douments First")
            st.stop()
        with st.spinner("Processing Documents"):
            doc = get_files_text(uploaded_doc)
            st.write("Files Uploaded ...")
            text_chunks = get_text_chunks(doc)
            st.write("Chunks Created Successfully")
            vector_store = get_vectorstore(text_chunks)
            st.write("Vectore Store Created...")
            st.write(vector_store)

        st.session_state.conversation = get_conversation_chain(vector_store,groq_api_key)
        st.session_state.processcomplete = True

    if  st.session_state.processcomplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            with st.spinner("Thinking"):
                handel_userinput(user_question)


def get_files_text(files):
    text=""
    for file in files:
        if file.name.endswith(".pdf"):
            text+=get_pdf_text(file)
        elif file.name.endswith(".docx"):
            text+=get_docx_text(file)
        elif file.name.endswith(".csv"):
            text+=get_csv_text(file)

    return text

def get_pdf_text(pdf_file):
    text=""
    pdf_reader=PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text

def get_docx_text(docxs):
    all_text = []
    doc = docx.Document(docxs)
    for para in doc.paragraphs:
        all_text.append(para.text)
    text=''.join(all_text)
    return text

def get_csv_text():
    return "a"

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator='\n', chunk_size=4096, chunk_overlap=300, length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Using the hugging face embedding models
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # creating the Vectore Store using Facebook AI Semantic search
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    
    return knowledge_base

def get_conversation_chain(vetorestore,api_keys):
    llm = ChatGroq(groq_api_key=api_keys, model_name = 'Llama3-70b-8192',temperature=0.5,)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory,  
    )
    return conversation_chain


def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__ == '__main__':
    main()