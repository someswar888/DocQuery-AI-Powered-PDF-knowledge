import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms import google_palm
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import faiss
import os

os.environ['GOOGLE_API_KEY']='AIzaSyAzsuiQSAmU1ncdmpInmkkhXymhe9DXmmo'

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        Pdf_Reader=PdfReader(pdf)
        for page in Pdf_Reader.pages:
            text+= page.extract_text()
            return text
        
def get_text_chunks(text):
    text_Splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks=text_Splitter.split_text(text)
    return chunks

def get_conversational_chain(vector_store):
    llm=google_palm()
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    Conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriver(), memory=memory)
    return Conversation_chain

def user_input(user_queston):
    response = st.session_state.conversation({'question':user_queston})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2==0:
            st.write("Human:",message.content)
        else:
            st.write("Bot:",message.content)

def main():
    st.set_page_config("DocQuery:AI-Powered PDF Knowledge Assistant")
    st.header("DocQuery: AI-Powered PDF Knoeledge Assistant")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
def get_vector_store(text_chunks):
  with st.sidebar:
    st.title("settings")
    st.subheader("Upload your Documents")
    pdf_docs = st.file_uploader("Upload your PDF File and Click on Process Button ",accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector_store)
            st.success("Done")

            if __name__ == "__main__":
                main()




                    
                
            
                         






