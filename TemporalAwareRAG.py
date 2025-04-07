import os
import re
import spacy
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize Spacy NLP
nlp = spacy.load("en_core_web_sm")

# --- Configuration ---
MAX_CHUNK_GAP = 1000
DEFAULT_YEAR = 0
TOP_K = 10

# --- Text Loading ---
def load_texts(folder_path):
    texts = {}
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                texts[file] = f.read()
    return texts

# --- Date-Based Chunking ---
def chunk_by_dates(text):
    chunks = []
    current_chunk = []
    current_year = DEFAULT_YEAR
    pos = 0
    year_matches = list(re.finditer(r'\b(1[0-9]{3}|20[0-9]{2})\b', text))

    for i, match in enumerate(year_matches):
        year = int(match.group())
        start = match.start()
        if current_year != DEFAULT_YEAR and (year != current_year or start - pos > MAX_CHUNK_GAP):
            chunks.append(create_chunk(current_chunk, current_year, pos, start))
            current_chunk = []
            pos = start

        current_year = year
        current_chunk.append(text[pos:match.end()])
        pos = match.end()

    if pos < len(text):
        current_chunk.append(text[pos:])
    if current_chunk:
        chunks.append(create_chunk(current_chunk, current_year, pos, len(text)))

    return chunks

def create_chunk(chunks, year, start, end):
    return {
        "text": "".join(chunks).strip(),
        "year": year,
        "start": start,
        "end": end
    }

# --- Entity Extraction ---
def extract_entities(text):
    doc = nlp(text)
    return {
        "people": list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON")),
        "locations": list(set(ent.text for ent in doc.ents if ent.label_ == "GPE"))
    }

# --- Document Processing ---
def process_documents(folder_path):
    texts = load_texts(folder_path)
    all_docs = []

    for filename, content in texts.items():
        chunks = chunk_by_dates(content)
        print(f"File '{filename}' -> {len(chunks)} chunks")
        for chunk in chunks:
            entities = extract_entities(chunk["text"])
            print(f"Entities in '{filename}': People={entities['people']} | Locations={entities['locations']}")
            metadata = {
                "source": filename,
                "year": chunk["year"],
                "people": entities["people"],
                "locations": entities["locations"]
            }
            all_docs.append(Document(
                page_content=chunk["text"],
                metadata=clean_metadata(metadata)
            ))
        print("\n=== Document Count by Source ===")
        from collections import Counter
        counter = Counter(doc.metadata['source'] for doc in all_docs)
        for src, count in counter.items():
            print(f"{src}: {count} chunks")

    return all_docs

def clean_metadata(meta):
    cleaned = {}
    for key, value in meta.items():
        if isinstance(value, list):
            cleaned[key] = ", ".join([v.lower() for v in value])
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

    print(f"Query parsed as: {parsed}")
    print(f"Filters applied: {filters}")

    results = []

    try:
        if filters:
            filtered_results = vectorstore.similarity_search(query, k=2, filter=filters)
            results.extend([(doc, "entity") for doc in filtered_results[:2]])
    except Exception as e:
        print(f"Filtered search error: {e}")

    try:
        semantic_results = vectorstore.similarity_search(query, k=2)
        results.extend([(doc, "semantic") for doc in semantic_results[:2]])
    except Exception as e:
        print(f"Semantic search error: {e}")

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
        filters["people"] = {"$contains": parsed["people"][0].lower()}
    if parsed["locations"]:
        filters["locations"] = {"$contains": parsed["locations"][0].lower()}

    return filters

# --- Main Execution ---
if __name__ == "__main__":
    DATA_DIR = "/home/akash/HistoriChat/data"
    home_directory = os.path.expanduser("~")
    PERSIST_DIR = os.path.join(home_directory, "Historichat")

    print("\n=== Path Verification ===")
    print(f"Data directory exists: {os.path.exists(DATA_DIR)}")
    print(f"Persist directory exists: {os.path.exists(PERSIST_DIR)}")

    if not os.path.exists(PERSIST_DIR):
        print("\n=== Processing Documents ===")
        docs = process_documents(DATA_DIR)
        print(f"Processed {len(docs)} documents")

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=PERSIST_DIR,
            collection_name="HC-2"
        )
        print("Created new vectorstore")
    else:
        print("\n=== Loading Existing Vectorstore ===")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )
        print(f"Existing collection contains {vectorstore._collection.count()} documents")

    print("\n=== Testing Query ===")
    test_query = "tell me about alfred the great"
    print("Running query:", test_query)
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
        print("\nNo results found for query")
