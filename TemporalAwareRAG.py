import os
import re
import spacy
from datetime import datetime
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# --- Load Texts ---
def load_text_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    texts = {}
    for file in files:
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            texts[file] = f.read()
    return texts

# --- Extract Years ---
def extract_years(text):
    return sorted({int(year) for year in re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)})

# --- Extract Entities ---
def extract_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "EVENT", "DATE", "GPE"}:
            entities[ent.label_].append(ent.text)
    return {k.lower(): list(set(v)) for k, v in entities.items()}

# --- Chunk Text by Timeline ---
def chunk_by_timeline(text, chunk_size=1000, max_gap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    raw_chunks = splitter.split_text(text)
    
    enriched_chunks, current_chunk, current_years = [], [], []

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

# --- Store in ChromaDB ---
def sanitize_metadata(metadata):
    """Sanitize metadata to ensure Chroma-compatible types."""
    safe_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            safe_metadata[k] = v
        elif isinstance(v, list):
            safe_metadata[k] = ", ".join(map(str, v))
        elif v is None:
            safe_metadata[k] = ""
        else:
            safe_metadata[k] = str(v)
    return safe_metadata

def process_and_store_in_chromadb(folder_path):
    texts = load_text_files(folder_path)
    all_documents = []

    for filename, text in texts.items():
        chunks = chunk_by_timeline(text)
        for i, chunk in enumerate(chunks):
            metadata = {
                "file": filename,
                "chunk_index": i,
                "start_year": chunk["start_year"],
                "end_year": chunk["end_year"],
                "years": chunk["years"],
                "person": extract_entities(chunk["text"]).get("person", []),
            }

            sanitized = sanitize_metadata(metadata)
            print("[DEBUG] Adding document with metadata:", sanitized)
            all_documents.append({"text": chunk["text"], "metadata": sanitized})

    persist_directory = os.path.expanduser("~/HistoriChat")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="HC-1"
    )

    for doc in all_documents:
        vectorstore.add_texts([doc["text"]], metadatas=[doc["metadata"]])

    print(f"[INFO] Stored {len(all_documents)} documents.")
    return vectorstore


# --- Parse Query ---
def parse_query(query):
    doc = nlp(query)
    entities = {"PERSON": [], "EVENT": [], "DATE": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    years = sorted(set(map(int, re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', query))))
    keywords = []
    if "before" in query.lower(): keywords.append("before")
    if "after" in query.lower(): keywords.append("after")
    if "during" in query.lower(): keywords.append("during")
    if any(k in query.lower() for k in ["first", "second", "third"]): keywords.append("ordered")
    return {"entities": entities, "years": years, "temporal_keywords": keywords}

# --- Main Retrieval ---
def retrieve_historical_facts(query):
    print(f"[INFO] Query: {query}")
    parsed = parse_query(query)
    print(f"[DEBUG] Parsed query: {parsed}")
    
    filters = []

    if parsed["entities"]["PERSON"]:
        name = parsed["entities"]["PERSON"][0].lower()
        filters.append({"person": {"$contains": name}})

    if parsed["years"]:
        year = parsed["years"][0]
        if "before" in parsed["temporal_keywords"]:
            filters.append({"start_year": {"$lt": year}})
        elif "after" in parsed["temporal_keywords"]:
            filters.append({"start_year": {"$gt": year}})
        elif "during" in parsed["temporal_keywords"]:
            filters.append({"$and": [{"start_year": {"$lte": year}}, {"end_year": {"$gte": year}}]})

    chroma = Chroma(
        persist_directory=os.path.expanduser("~/HistoriChat"),
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="HC-1"
    )

    search_kwargs = {"k": 10}
    if filters:
        if len(filters) == 1:
            search_kwargs["filter"] = filters[0]
        else:
            search_kwargs["filter"] = {"$and": filters}
        print(f"[DEBUG] Using filter: {search_kwargs['filter']}")

    retriever = chroma.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    results = retriever.invoke(query)

    if not results:
        print("[WARN] No results with filter. Retrying without filters...")
        retriever = chroma.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        results = retriever.invoke(query)
        print(f"[DEBUG] Retrieved {len(results)} results after fallback")

    if "ordered" in parsed["temporal_keywords"]:
        results = sorted(results, key=lambda doc: doc.metadata.get("start_year", 9999))

    return results

# --- Debug Helper: Print All Stored Chunks ---
def debug_show_all_documents():
    chroma = Chroma(
        persist_directory=os.path.expanduser("~/HistoriChat"),
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="HC-1"
    )
    retriever = chroma.as_retriever(search_kwargs={"k": 50})
    results = retriever.invoke("history")
    print(f"[INFO] Total retrieved (debug): {len(results)}")
    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1} ---")
        print("Metadata:", doc.metadata)
        print("Text:", doc.page_content[:250])

# --- Main Execution ---
if __name__ == "__main__":
    folder_path = "/home/akash/HistoriChat/data"

    # Step 1: Ingest and store
    process_and_store_in_chromadb(folder_path)

    # Step 2: Query
    query = "What did William do before 1066?"
    results = retrieve_historical_facts(query)

    print(f"\n[RESULT] Results for query: '{query}'\n")
    if not results:
        print("No documents found.")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Year Range: {doc.metadata.get('start_year')} - {doc.metadata.get('end_year')}")
        print(f"Text: {doc.page_content[:300]}...\n")
