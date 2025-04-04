import os
import re
import uuid
import glob
from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
import spacy
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize text splitter and embedding model
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma DB collection
vectorstore = Chroma(collection_name="historical_docs", embedding_function=embedding_model)

# Utility to extract named entities from a text chunk
def extract_metadata(text: str) -> Dict[str, Any]:
    doc = nlp(text)
    entities = {"PERSON": [], "DATE": [], "EVENT": [], "GPE": []}
    years = []

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        if ent.label_ == "DATE":
            found_years = re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", ent.text)
            years.extend([int(y) for y in found_years])

    start_year = min(years) if years else -1
    end_year = max(years) if years else -1

    # Avoid None in metadata
    metadata = {
        "person": [p.lower() for p in entities["PERSON"]] or [],
        "date": entities["DATE"] or [],
        "event": entities["EVENT"] or [],
        "gpe": entities["GPE"] or [],
        "years": years or [],
        "start_year": start_year if start_year != -1 else 0,
        "end_year": end_year if end_year != -1 else 0
    }

    return metadata

# Preprocess and store documents
def process_and_store_in_chromadb(folder_path: str):
    all_docs = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        loader = TextLoader(file_path)
        raw_docs = loader.load()
        for doc in raw_docs:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = extract_metadata(chunk)
                metadata.update({
                    "file": os.path.basename(file_path),
                    "chunk_index": i
                })
                # Debug metadata
                logging.debug(f"Adding document with metadata: {metadata}")
                try:
                    filtered = filter_complex_metadata(metadata)
                    all_docs.append(Document(page_content=chunk, metadata=filtered))
                except Exception as e:
                    logging.warning(f"Skipping chunk due to metadata issue: {e}")
    vectorstore.add_documents(all_docs)
    logging.info(f"Stored {len(all_docs)} documents.")

# Retrieve documents based on temporal + entity-aware query
def retrieve_historical_facts(query: str, top_k: int = 5):
    doc = nlp(query)
    entities = {"PERSON": [], "EVENT": [], "DATE": [], "GPE": []}
    years = []
    temporal_keywords = []

    for token in doc:
        if token.text.lower() in ["before", "after", "between", "during"]:
            temporal_keywords.append(token.text.lower())

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        if ent.label_ == "DATE":
            years += [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", ent.text)]

    logging.debug(f"Parsed query: {{'entities': {entities}, 'years': {years}, 'temporal_keywords': {temporal_keywords}}}")

    filters = []
    person = [p.lower() for p in entities["PERSON"]]

    if person:
        filters.append({"person": {"$in": person}})
    if "before" in temporal_keywords and years:
        filters.append({"start_year": {"$lt": years[0]}})
    elif "after" in temporal_keywords and years:
        filters.append({"end_year": {"$gt": years[0]}})
    elif "between" in temporal_keywords and len(years) >= 2:
        filters.append({"start_year": {"$gte": years[0]}})
        filters.append({"end_year": {"$lte": years[1]}})

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    try:
        if filters:
            logging.debug(f"Using filter: {{'$and': {filters}}}")
            results = vectorstore.similarity_search(query, filter={"$and": filters}, k=top_k)
            if not results:
                raise ValueError("No results with filters")
        else:
            results = vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        logging.warning(f"No results with filter. Retrying without filters... ({e})")
        results = vectorstore.similarity_search(query, k=top_k)

    logging.info(f"Results for query: '{query}'")
    for doc in results:
        print(doc.page_content[:300] + "...\n")
    return results

# Run pipeline
if __name__ == "__main__":
    folder_path = "/home/akash/HistoriChat/data" 
    process_and_store_in_chromadb(folder_path)

    query = "What did William do before 1066?"
    retrieve_historical_facts(query)
