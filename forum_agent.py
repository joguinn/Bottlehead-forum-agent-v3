# forum_agent.py

import os
import urllib.request
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Dropbox links to FAISS files (use ?dl=1 or &raw=1 to force download)
FAISS_URL = "https://www.dropbox.com/scl/fi/koh76ewpxjcijmpsoj3j3/index.faiss?rlkey=mbzhm4583c2w9kajrdl5hxswx&dl=1"
PKL_URL = "https://www.dropbox.com/scl/fi/0kxoaaxuu7joy1hz3o8ja/index.pkl?rlkey=e792h3eryayafz2g8elv3f1lo&dl=1"

# Ensure faiss_index/ folder exists
os.makedirs("faiss_index", exist_ok=True)

# Download FAISS index files from Dropbox if missing
def download_faiss_index():
    if not os.path.exists("faiss_index/index.faiss"):
        st.info("⬇️ Downloading FAISS index...")
        urllib.request.urlretrieve(FAISS_URL, "faiss_index/index.faiss")
    if not os.path.exists("faiss_index/index.pkl"):
        st.info("⬇️ Downloading FAISS metadata...")
        urllib.request.urlretrieve(PKL_URL, "faiss_index/index.pkl")

# Load OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set up LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

@st.cache_resource
def load_agent():
    try:
        download_faiss_index()
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        st.error(f"❌ Error loading FAISS index: {e}")
        return None

forum_agent = load_agent()

def ask_forum_agent(query, product=None):
    if not forum_agent:
        return "⚠️ Forum agent is not available."

    if product:
        retriever = forum_agent.retriever
        retriever.search_kwargs["filter"] = lambda doc: isinstance(doc.metadata, dict) and doc.metadata.get("product", "").lower() == product.lower()

    return forum_agent.run(query)
