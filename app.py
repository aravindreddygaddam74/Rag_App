import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from secret_api_keys import OPENAI_API_KEY

# Load OpenAI LLM
def load_llm():
    return ChatOpenAI(
        temperature=0.5,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo"
    )

# Extract text from PDF
def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Extract text from .txt file
def extract_text_from_txt(uploaded_txt):
    return uploaded_txt.read().decode("utf-8")

# Extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        return response.text
    except Exception as e:
        return f"Failed to fetch URL: {e}"

# UI title
st.title("🧠 RAG QA App (PDF, Text, URL, etc.)")

# Input type selection
input_mode = st.selectbox("Select Input Type", ["PDF File", "Text File", "Paste Text", "Web URL"])
text = ""

# Content ingestion
if input_mode == "PDF File":
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_pdf and st.button("Proceed"):
        text = extract_text_from_pdf(uploaded_pdf)

elif input_mode == "Text File":
    uploaded_txt = st.file_uploader("Upload Text File", type="txt")
    if uploaded_txt and st.button("Proceed"):
        text = extract_text_from_txt(uploaded_txt)

elif input_mode == "Paste Text":
    pasted_text = st.text_area("Paste your content here")
    if pasted_text and st.button("Proceed"):
        text = pasted_text

elif input_mode == "Web URL":
    url = st.text_input("Enter webpage URL")
    if url and st.button("Proceed"):
        text = extract_text_from_url(url)

# After content is loaded
if text:
    try:
        # Chunking
        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        # Embedding & vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        st.session_state.qa_chain = qa_chain
        st.success("✅ Content processed! Ask your question below.")

    except Exception as e:
        st.error(f"Error during processing: {e}")

# Ask a question
question = st.text_input("Ask a question from the content")
if question and "qa_chain" in st.session_state:
    try:
        response = st.session_state.qa_chain({"query": question})
        answer = response["result"]
        sources = response["source_documents"]

        st.markdown(f"### 🧠 Answer:\n{answer}")

        with st.expander("📚 Source Snippets"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}:**\n```{doc.page_content[:500]}```")

    except Exception as e:
        st.error(f"Error during question answering: {e}")
