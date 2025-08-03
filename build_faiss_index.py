# build_faiss_index.py
# One-time script to build and save your FAISS index from the Q&A CSV

import pandas as pd
import os

import pandas as pd
from getpass import getpass

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Set your OpenAI API key here (or via environment variable)
from getpass import getpass
openai_api_key = getpass("🔑 Enter your OpenAI API key: ")

# Load CSV

# Prompt for OpenAI API key
openai_api_key = getpass("🔑 Enter your OpenAI API key: ")

# Load your forum Q&A CSV

import os
import pandas as pd

DROPBOX_CSV_URL = "https://www.dropbox.com/scl/fi/e5v4hxp68apt7n13hits6/bottlehead_qna.csv?rlkey=ly6wrdcyobqqj4ug2gr9cin61&dl=1"

def load_data():
    if os.path.exists("bottlehead_qna.csv"):
        print("🔍 Loading local CSV...")
        df = pd.read_csv("bottlehead_qna.csv")
    else:
        print("🌐 Downloading CSV from Dropbox...")
        df = pd.read_csv(DROPBOX_CSV_URL)
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()


# Validate required columns
if "question" not in df.columns or "answer" not in df.columns:
    raise ValueError("CSV must contain 'question' and 'answer' columns.")

# Convert to LangChain Documents

# Check required columns
if "question" not in df.columns or "answer" not in df.columns:
    raise ValueError("CSV must contain 'question' and 'answer' columns.")

# Build LangChain Documents

docs = []
for i, row in df.iterrows():
    q = str(row["question"]).strip()
    a = str(row["answer"]).strip()
    metadata = {}
    if "product" in df.columns and pd.notna(row["product"]):
        metadata["product"] = row["product"].strip()
    content = f"Q: {q}\nA: {a}"
    docs.append(Document(page_content=content, metadata=metadata))


# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Embed and store in FAISS
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
faiss_index = FAISS.from_documents(split_docs, embeddings)

# Save index to disk
faiss_index.save_local("faiss_index")
print("✅ FAISS index saved to ./faiss_index")

# Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Embed and create FAISS index
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
faiss_index = FAISS.from_documents(split_docs, embeddings)

# Save index to local folder
faiss_index.save_local("faiss_index")
print("✅ FAISS index saved to ./faiss_index")
