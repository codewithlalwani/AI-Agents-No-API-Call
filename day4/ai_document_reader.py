import streamlit as st
import faiss
import numpy as np
import PyPDF2
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

#load ai model 
llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")

#load huggingface embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#initialize FAISS vector database
index = faiss.IndexFlatL2(384) # vector dimension for miniLM
vector_store = {}

def extract_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

#function to store text in FAISS vector database
def store_in_faiss(text, file_name):
    global index, vector_store
    st.write("Storing text in FAISS vector database...")

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)

    # convert text into embeddings
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    # store in FAISS index
    index.add(vectors)
    vector_store[len(vector_store)] = (file_name, texts)  # store file name and texts for retrieval

    return "Document stored successfully!"


# function to retrieve relevant text from FAISS vector database
def retrieve_from_faiss(query):
    global index, vector_store
    st.write("Retrieving relevant text from FAISS vector database...")

    # convert query into embedding
    query_vector = embeddings.embed_query(query)
    query_vector = np.array([query_vector], dtype=np.float32).reshape(1, -1)

    # search in FAISS indexx
    D, I = index.search(query_vector, k=2)  # retrieve top 5 results

    context = ""
    for idx in I[0]:
        if idx in vector_store:
            context += " ".join(vector_store[idx][1]) + "\n"

    if not context:
        return "No relevant context found in the database."

    # ask AI to generate an answer
    return llm.invoke(f"Answer the following question based on the context: {context}\nQuestion: {query}")

retrieve_and_answer = retrieve_from_faiss

# Streamlit app
st.title("AI Document Reader")
st.write("Upload a PDF document and ask questions about its content.")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:
    text = extract_from_pdf(uploaded_file)
    # Store in FAISS
    store_message = store_in_faiss(text, uploaded_file.name)
    st.write(store_message)

# user input for q&a
query = st.text_input("Ask a question about the document:")
if query:
    answer = retrieve_and_answer(query)
    st.subheader("Answer:")
    st.write(answer)






