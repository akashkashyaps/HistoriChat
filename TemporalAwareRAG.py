import os
import re
import shutil
import spacy
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize Spacy NLP
nlp = spacy.load("en_core_web_sm")

# --- Configuration ---
DEFAULT_YEAR = 0
TOP_K = 10
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Text Loading ---
def load_texts(folder_path):
    """Load all .txt files from a directory"""
    texts = {}
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                texts[file] = f.read()
    return texts

# --- Year Extraction ---
def extract_first_year(text):
    match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
    return int(match.group()) if match else DEFAULT_YEAR

# --- Entity Extraction ---
def extract_entities(text):
    doc = nlp(text)
    return {
        "people": list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON")),
        "locations": list(set(ent.text for ent in doc.ents if ent.label_ == "GPE"))
    }

# --- Document Processing ---
def process_documents(folder_path):
    """Main processing pipeline"""
    texts = load_texts(folder_path)

    print("\n=== Loaded Text Files ===")
    for name, content in texts.items():
        print(f"{name}: {len(content)} characters")
    print(f"Total files loaded: {len(texts)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_docs = []

    for filename, content in texts.items():
        chunks = splitter.split_text(content)
        for chunk in chunks:
            entities = extract_entities(chunk)
            year = extract_first_year(chunk)

            metadata = {
                "source": filename,
                "year": year,
                "people": entities["people"],
                "locations": entities["locations"]
            }

            all_docs.append(Document(
                page_content=chunk,
                metadata=clean_metadata(metadata)
            ))

    return all_docs

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

def retrieve(query, vectorstore):
    parsed = parse_query(query)
    filters = build_filters(parsed)
    results = []

    # Entity-based search
    try:
        if filters:
            filtered_results = vectorstore.similarity_search(query, k=2, filter=filters)
            results.extend([(doc, "entity") for doc in filtered_results])
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

# --- Main Execution ---
if __name__ == "__main__":
    DATA_DIR = "/home/akash/HistoriChat/data"
    home_directory = os.path.expanduser("~")
    PERSIST_DIR = os.path.join(home_directory, "Historichat")

    print("\n=== Path Verification ===")
    print(f"Data directory exists: {os.path.exists(DATA_DIR)}")

    # Always delete existing vectorstore
    if os.path.exists(PERSIST_DIR):
        print("Deleting old vectorstore...")
        shutil.rmtree(PERSIST_DIR)

    print("\n=== Processing Documents ===")
    docs = process_documents(DATA_DIR)
    print(f"Total chunks created: {len(docs)}")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=PERSIST_DIR,
        collection_name="HC-2"
    )
    print("Vectorstore created and saved.")

    # Test Query
    print("\n=== Testing Query ===")
    test_query = "tell me about alfred the great"
    print("Query:", test_query)
    results = retrieve(test_query, vectorstore)
    print(f"Found {len(results)} unique results")

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
