import os
import re
import spacy
from collections import Counter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize Spacy
nlp = spacy.load("en_core_web_sm")

# --- Text Loading ---
def load_texts(folder_path):
    texts = {}
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                texts[file] = f.read()
    return texts

# --- Metadata Extraction ---
def extract_metadata(text):
    doc = nlp(text)
    people = list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON"))
    locations = list(set(ent.text for ent in doc.ents if ent.label_ == "GPE"))
    years = [int(y) for y in re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)]
    first_year = years[0] if years else 0
    return {
        "people": people,
        "locations": locations,
        "year": first_year
    }

def clean_metadata(meta):
    cleaned = {}
    for key, value in meta.items():
        if isinstance(value, list):
            cleaned[key] = ", ".join(value)
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned

# --- Process Documents ---
def process_documents(folder_path):
    texts = load_texts(folder_path)
    all_docs = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filename, content in texts.items():
        metadata = extract_metadata(content)
        metadata["source"] = filename
        cleaned = clean_metadata(metadata)
        splits = splitter.split_text(content)

        for split in splits:
            all_docs.append(Document(page_content=split, metadata=cleaned))

    print("\n=== Document Count by Source ===")
    counter = Counter(doc.metadata['source'] for doc in all_docs)
    for src, count in counter.items():
        print(f"{src}: {count} chunks")

    return all_docs

# --- Query Handling ---
def parse_query(query):
    doc = nlp(query)
    return {
        "people": list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON")),
        "locations": list(set(ent.text for ent in doc.ents if ent.label_ == "GPE")),
        "years": [int(y) for y in re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', query)],
        "temporal_keywords": {
            "before": "before" in query.lower(),
            "after": "after" in query.lower(),
            "during": "during" in query.lower()
        }
    }

def build_filters(parsed):
    filters = {}
    if parsed["years"]:
        year = parsed["years"][0]
        if parsed["temporal_keywords"]["before"]:
            filters["year"] = {"$lt": year}
        elif parsed["temporal_keywords"]["after"]:
            filters["year"] = {"$gt": year}
        else:
            filters["year"] = year
    if parsed["people"]:
        filters["people"] = {"$contains": parsed["people"][0]}
    if parsed["locations"]:
        filters["locations"] = {"$contains": parsed["locations"][0]}
    return filters

def retrieve(query, vectorstore):
    parsed = parse_query(query)
    filters = build_filters(parsed)

    results = []

    # Entity-based filtered search
    try:
        if filters:
            entity_results = vectorstore.similarity_search(query, k=2, filter=filters)
            results.extend([(doc, "entity") for doc in entity_results])
    except Exception as e:
        print(f"Entity search error: {e}")

    # Semantic search
    try:
        semantic_results = vectorstore.similarity_search(query, k=2)
        results.extend([(doc, "semantic") for doc in semantic_results])
    except Exception as e:
        print(f"Semantic search error: {e}")

    # Deduplicate
    seen = set()
    unique_results = []
    for doc, label in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append((doc, label))

    return unique_results

# --- Main Execution ---
if __name__ == "__main__":
    DATA_DIR = "/home/akash/HistoriChat/data"
    home_directory = os.path.expanduser("~")
    PERSIST_DIR = os.path.join(home_directory, "Historichat")

    print("\n=== Path Verification ===")
    print(f"Data directory exists: {os.path.exists(DATA_DIR)}")
    print(f"Persist directory exists: {os.path.exists(PERSIST_DIR)}")

    if not os.path.exists(PERSIST_DIR):
        print("\n=== Processing and Building Vectorstore ===")
        docs = process_documents(DATA_DIR)
        print(f"Total chunks: {len(docs)}")

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=PERSIST_DIR,
            collection_name="HC-2"
        )
        print("New vectorstore created.")
    else:
        print("\n=== Loading Existing Vectorstore ===")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )
        print(f"Existing collection contains {vectorstore._collection.count()} documents")

    test_query = "Tell me about Alfred the Great"
    print("\n=== Testing Query ===")
    results = retrieve(test_query, vectorstore)
    print(f"Found {len(results)} results")

    if results:
        print("\n=== Results ===")
        for i, (doc, label) in enumerate(results):
            print(f"\nResult {i+1} ({label.upper()}):")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Year: {doc.metadata.get('year', 'N/A')}")
            print(f"People: {doc.metadata.get('people', 'N/A')}")
            print(f"Text: {doc.page_content[:300]}...")
    else:
        print("No results found.")
