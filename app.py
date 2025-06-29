import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from processa_pdf import carregar_e_indexar_documento

st.set_page_config(page_title="Assistente PDF LLM", layout="centered")

st.title("📄🤖 Assistente de PDFs com LLM")

openai_key = st.text_input("🔑 OpenAI API Key", type="password")

uploaded_file = st.file_uploader("📎 Faça upload de um PDF", type="pdf")

if uploaded_file and openai_key:
    with open(f"docs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("Documento carregado!")

    vectorstore = carregar_e_indexar_documento(f"docs/{uploaded_file.name}", openai_key)
    llm = OpenAI(temperature=0.5, openai_api_key=openai_key)
    chain = load_qa_chain(llm, chain_type="stuff")

    st.subheader("💬 Pergunte algo sobre o documento")

    pergunta = st.text_input("Digite sua pergunta")

    if pergunta:
        docs = vectorstore.similarity_search(pergunta)
        resposta = chain.run(input_documents=docs, question=pergunta)
        st.markdown(f"**Resposta:** {resposta}")
