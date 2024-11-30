from pathlib import Path
import json
import shutil
import glob
import os
from tqdm import tqdm
import time
import torch

import chromadb
from chromadb.config import Settings

from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from sentence_transformers import SentenceTransformer, util

# Check if CUDA or MPS is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Timer function to measure execution time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper

# Load data from the specified text file
@timer
def load_data(file_path):
    loader = TextLoader(file_path, autodetect_encoding=True)
    data = loader.load()
    return data

# Splitting loaded documents into smaller chunks for better processing
@timer
def split_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", "."]
    )
    all_splits = text_splitter.split_documents(data)
    return all_splits

# Specifying the model for generating embeddings and loading it
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": device}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Remove the existing Chroma database directory if it exists
shutil.rmtree("chroma_db", ignore_errors=True)

# Create a Chroma vector database from the document splits using the specified embeddings
@timer
def create_vector_db(all_splits):
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
    return vectordb

# Retrieve documents relevant to the query using Maximal Marginal Relevance (MMR) for diversity
@timer
def retrieve_documents(query, vectordb):
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    return docs

# Base directories for the input and output files
input_dir = os.path.expanduser('~/Desktop/Internship/extracted_bios_1/')
content_dir = os.path.expanduser('~/Desktop/Internship/Test Content/')
output_dir = os.path.expanduser('~/Desktop/Internship/Others_extracted_content/')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file in the input directory
for file_path in glob.glob(os.path.join(input_dir, '*.txt')):
    # Extract the QID from the filename
    qid = Path(file_path).stem

    # Load the text file data
    data, load_time = load_data(file_path)
    print(f"Data loading for {qid} completed in {load_time:.2f} seconds")

    # Split the document into chunks
    all_splits, split_time = split_documents(data)
    print(f"Document splitting for {qid} completed in {split_time:.2f} seconds")

    # Create the Chroma vector database
    vectordb, db_creation_time = create_vector_db(all_splits)
    print(f"Chroma vector database creation for {qid} completed in {db_creation_time:.2f} seconds")

    # Load the corresponding content file
    content_file_path = os.path.join(content_dir, f'{qid}_content.json')
    with open(content_file_path, 'r') as file:
        qid_content = json.load(file)

    # Process each key in the content file and retrieve document chunks
    extracted_content = {}
    for key, query_text in qid_content.items():
        docs, retrieval_time = retrieve_documents(query_text, vectordb)
        print(f"Document retrieval for '{key}' in {qid} completed in {retrieval_time:.2f} seconds")

        # Collect the retrieved document chunks as a list
        retrieved_chunks = [doc.page_content for doc in docs]
        extracted_content[key] = retrieved_chunks

    # Save the extracted content to a JSON file
    output_file_path = os.path.join(output_dir, f'{qid}_extracted.json')
    with open(output_file_path, 'w') as outfile:
        json.dump(extracted_content, outfile, indent=4)

    print(f"Extracted content for {qid} saved to {output_file_path}")
