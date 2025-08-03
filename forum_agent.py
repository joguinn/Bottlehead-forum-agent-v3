# forum_agent.py

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

# Load FAISS index
@st.cache_resource
def load_agent():
    try:
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
        # Filter documents based on metadata
        retriever.search_kwargs["filter"] = lambda doc: isinstance(doc.metadata, dict) and doc.metadata.get("product", "").lower() == product.lower()

    return forum_agent.run(query)
