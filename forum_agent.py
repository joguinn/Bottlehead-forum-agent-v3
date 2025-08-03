# forum_agent.py
import os
import urllib.request
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings

# --- Configuration ---
FAISS_DIR = "faiss_index"
FAISS_INDEX_URL = "https://www.dropbox.com/scl/fi/koh76ewpxjcijmpsoj3j3/index.faiss?rlkey=mbzhm4583c2w9kajrdl5hxswx&dl=1"
FAISS_PKL_URL = "https://www.dropbox.com/scl/fi/0kxoaaxuu7joy1hz3o8ja/index.pkl?rlkey=e792h3eryayafz2g8elv3f1lo&dl=1"

# --- Download FAISS index if missing ---
def download_faiss():
    os.makedirs(FAISS_DIR, exist_ok=True)
    urllib.request.urlretrieve(FAISS_INDEX_URL, f"{FAISS_DIR}/index.faiss")
    urllib.request.urlretrieve(FAISS_PKL_URL, f"{FAISS_DIR}/index.pkl")

if not os.path.exists(f"{FAISS_DIR}/index.faiss") or not os.path.exists(f"{FAISS_DIR}/index.pkl"):
    print("📥 Downloading FAISS index from Dropbox...")
    download_faiss()

# --- Load OpenAI API key ---
openai_api_key = st.secrets["OPENAI_API_KEY"]

# --- Set up LLM ---
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

# --- Load FAISS index ---
@st.cache_resource
def load_agent():
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.load_local(FAISS_DIR, embeddings=embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        st.error(f"❌ Error loading FAISS index: {e}")
        return None

forum_agent = load_agent()

# --- Ask the agent ---
def ask_forum_agent(query, product=None):
    if not forum_agent:
        return "⚠️ Forum agent is not available."

    if product:
        retriever = forum_agent.retriever
        retriever.search_kwargs["filter"] = lambda doc: doc.metadata.get("product", "").lower() == product.lower()

    return forum_agent.run(query)