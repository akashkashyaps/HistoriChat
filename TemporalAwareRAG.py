import os
import re
import spacy
from datetime import datetime
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import Document

# Load NLP model
nlp = spacy.load("en_core_web_sm")

folder_path = "/home/akash/HistoriChat/data"

def load_text_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    texts = {}
    for file in files:
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            texts[file] = f.read()
    return texts

def extract_years(text):
    return sorted({int(y) for y in re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)})

def extract_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "EVENT", "DATE", "GPE"}:
            entities[ent.label_].append(ent.text)
    for key in entities:
        entities[key] = list(set(entities[key]))
    return dict(entities)

def chunk_by_timeline(text, chunk_size=1000, max_gap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    raw_chunks = splitter.split_text(text)

    enriched_chunks = []
    current_chunk = []
    current_years = []

    for chunk in raw_chunks:
        years = extract_years(chunk)

        if not current_chunk:
            current_chunk.append(chunk)
            current_years.extend(years)
            continue

        if years and current_years and (min(years) - max(current_years) <= max_gap):
            current_chunk.append(chunk)
            current_years.extend(years)
        else:
            enriched_chunks.append({
                "text": " ".join(current_chunk),
                "years": sorted(set(current_years)),
                "start_year": min(current_years) if current_years else None,
                "end_year": max(current_years) if current_years else None
            })
            current_chunk = [chunk]
            current_years = years

    if current_chunk:
        enriched_chunks.append({
            "text": " ".join(current_chunk),
            "years": sorted(set(current_years)),
            "start_year": min(current_years) if current_years else None,
            "end_year": max(current_years) if current_years else None
        })

    return enriched_chunks

def process_and_store_in_chromadb(folder_path):
    texts = load_text_files(folder_path)
    all_documents = []

    for filename, text in texts.items():
        chunks = chunk_by_timeline(text)
        for i, chunk in enumerate(chunks):
            entities = extract_entities(chunk["text"])
            metadata = {
                "file": filename,
                "chunk_index": i,
                "start_year": chunk["start_year"],
                "end_year": chunk["end_year"],
                "years": chunk["years"],
                "person": entities.get("PERSON", []),
                "event": entities.get("EVENT", []),
                "location": entities.get("GPE", []),
            }
            all_documents.append({"text": chunk["text"], "metadata": metadata})

    persist_directory = os.path.expanduser("~/HistoriChat")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="HC-1"
    )

    for doc in all_documents:
        vectorstore.add_texts([doc["text"]], metadatas=[doc["metadata"]])

    print(f"[INFO] Stored {len(all_documents)} chunks in ChromaDB.")
    return vectorstore

# --- Query Parsing ---

def parse_query(query):
    doc = nlp(query)
    entities = {"PERSON": [], "EVENT": [], "DATE": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    years = sorted(set(map(int, re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', query))))
    temporal_keywords = []

    if "before" in query.lower(): temporal_keywords.append("before")
    if "after" in query.lower(): temporal_keywords.append("after")
    if "during" in query.lower(): temporal_keywords.append("during")
    if any(w in query.lower() for w in ["first", "second", "third"]):
        temporal_keywords.append("ordered")

    return {
        "entities": entities,
        "years": years,
        "temporal_keywords": temporal_keywords
    }

# --- Retrieval with Fallback ---

def retrieve_historical_facts(query):
    parsed = parse_query(query)
    print("[DEBUG] Parsed query:", parsed)

    filters = []

    if parsed["entities"]["PERSON"]:
        filters.append({"person": {"$in": parsed["entities"]["PERSON"]}})
    if parsed["entities"]["EVENT"]:
        filters.append({"event": {"$in": parsed["entities"]["EVENT"]}})

    if parsed["years"]:
        year = parsed["years"][0]
        if "before" in parsed["temporal_keywords"]:
            filters.append({"start_year": {"$lt": year}})
        elif "after" in parsed["temporal_keywords"]:
            filters.append({"start_year": {"$gt": year}})
        elif "during" in parsed["temporal_keywords"]:
            filters.append({
                "$and": [
                    {"start_year": {"$lte": year}},
                    {"end_year": {"$gte": year}}
                ]
            })

    search_filter = {"$and": filters} if filters else None
    print("[DEBUG] Using filter:", search_filter)

    persist_directory = os.path.expanduser("~/HistoriChat")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="HC-1"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "filter": search_filter}
    )

    try:
        results = retriever.invoke(query)
        if not results:
            print("[WARN] No results with filter. Retrying without filters...")
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            results = retriever.invoke(query)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        results = retriever.invoke(query)

    if "ordered" in parsed["temporal_keywords"]:
        results = sorted(results, key=lambda doc: doc.metadata.get("start_year", 9999))

    return results

# --- Main Execution ---

if __name__ == "__main__":
    query = "What did William do before 1066?"
    print("[INFO] Query:", query)

    # If data isn’t already indexed, run this first
    # vectorstore = process_and_store_in_chromadb(folder_path)

    retrieved_docs = retrieve_historical_facts(query)

    print(f"\n[RESULT] Results for query: '{query}'\n")
    if not retrieved_docs:
        print("No documents found.")
    else:
        for doc in retrieved_docs:
            print(f"Year Range: {doc.metadata.get('start_year')} - {doc.metadata.get('end_year')}")
            print(f"Persons: {doc.metadata.get('person')}")
            print(f"Text: {doc.page_content[:300]}...\n")
