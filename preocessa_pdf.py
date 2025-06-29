import os
import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

def extrair_texto_pdf(caminho_pdf):
    texto = ""
    with open(caminho_pdf, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            texto += page.extract_text() or ""
    return texto

def carregar_e_indexar_documento(caminho_pdf, openai_api_key):
    texto = extrair_texto_pdf(caminho_pdf)
    
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentos = splitter.split_text(texto)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(documentos, embeddings)
    
    return vectorstore
