import requests
from bs4 import BeautifulSoup
import streamlit as st
import faiss
from langchain_ollama import OllamaLLM
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Load AI Model
llm = OllamaLLM(model="mistral")  # Change to "ll

#load hugging face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Change to another model if needed

