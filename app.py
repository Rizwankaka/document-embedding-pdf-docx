import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer


openapi_key = st.secrets["OPENAI_API_KEY"]

# "with" notation
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")
    st.header("📖 DocumentGPT 🤖")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf', 'docx'],accept_multiple_files=True)
        openai_api_key = openapi_key
        openai_api_key = st.text_input("OpenAI API Key", key=openapi_key , type="password")
        process = st.button("Process")
        # Setting Footer
        footer_html = """
            <div style="text-align: left;">
                <p style="font-size: 15px;"><b>Author: 🌟 Rizwan Rizwan 🌟</b></p>
                <a href="https://github.com/Rizwankaka"><img src="https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github" alt="GitHub"/></a><br>
                <a href="https://www.linkedin.com/in/rizwan-rizwan-1351a650/"><img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/></a><br>
                <a href="https://twitter.com/RizwanRizwan_"><img src="https://img.shields.io/badge/Twitter-Profile-blue?style=for-the-badge&logo=twitter" alt="Twitter"/></a><br>
                <a href="https://www.facebook.com/RIZWANNAZEEER"><img src="https://img.shields.io/badge/Facebook-Profile-blue?style=for-the-badge&logo=facebook" alt="Facebook"/></a><br>
                <a href="mailto:riwan.rewala@gmail.com"><img src="https://img.shields.io/badge/Gmail-Contact%20Me-red?style=for-the-badge&logo=gmail" alt="Gmail"/></a>
            </div>
            """
        st.markdown(footer_html, unsafe_allow_html=True)
        
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        st.write("file chunks created...")
        # create vetore stores
        vetor_store = get_vectorstore(text_chunks)
        st.write("Vectore Store Created...")
         # create conversation chain
        st.session_state.conversation = get_conversation_chain(vetor_store,openai_api_key) #for openAI

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handel_userinput(user_question)

# Function to get the input file and read the text from it.
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text

# Function to read PDF Files
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to read docx Files
def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

# Function to read csv Files
def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    # spilit ito chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Get selected embedding model
def get_vectorstore(text_chunks):
    # Using the hugging face embedding models ]

    # 1. all-MiniLM-L6-v2
    # 2. sentence-transformers/all-mpnet-base-v2 
    # 3. intfloat/e5-small-v2
    # 4. thenlper/gte-small 
    # 5. sentence-transformers/paraphrase-MiniLM-L6-v2 
     
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # List of embedding models to add - from here added 
    embeddings_models = [
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2", 
        "intfloat/e5-small-v2",
        "thenlper/gte-small",
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ]

    # Create dictionary to store embeddings
    all_embeddings = {} 

    for model_name in embeddings_models:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        all_embeddings[model_name] = embeddings

    # Allow user to select model   
    selected_model = st.sidebar.selectbox("Select embedding model", list(all_embeddings.keys()))

    # Get selected embedding model
    embeddings = all_embeddings[selected_model] ## -added
  
    # creating the Vector Store using Facebook AI Semantic search
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)

    return knowledge_base


def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
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
# Set a background image
def set_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.pexels.com/photos/6847584/pexels-photo-6847584.jpeg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image()

# Set a background image for the sidebar
sidebar_background_image = '''
<style>
[data-testid="stSidebar"] {
    background-image: url("https://images.pexels.com/photos/6101958/pexels-photo-6101958.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
    background-size: cover;
}
</style>
'''

st.sidebar.markdown(sidebar_background_image, unsafe_allow_html=True)
