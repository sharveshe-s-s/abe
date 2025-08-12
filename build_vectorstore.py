# backend/build_vectorstore.py

import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Load your CSV data
csv_files = [
    "data/agriloan.csv",
    "data/cropinfo.csv",
    "data/mandi.csv",
    "data/Pesticides.csv",
    "data/rainfall.csv",
    "data/region_wise_crop.csv",
    "data/weather.csv"
    # Add more CSV files here
]

documents = []
for file in csv_files:
    df = pd.read_csv(file)
    text = df.to_string(index=False)
    documents.append(Document(page_content=text))

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Initialize the multilingual embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create FAISS vector store
db = FAISS.from_documents(docs, embedding)

# Save the FAISS index
db.save_local("backend/vectorstore/faiss_index")
print("✅ FAISS index built and saved successfully.")
