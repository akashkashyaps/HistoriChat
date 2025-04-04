import os
import re
import spacy
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize Spacy for query parsing
nlp = spacy.load("en_core_web_sm")

# --- Configuration ---
MAX_CHUNK_GAP = 1000
DEFAULT_YEAR = 0
TOP_K = 5
FIXED_CHUNK_SIZE = 800  # For texts without years
CHUNK_OVERLAP = 200     # Overlap between chunks

# --- Text Loading ---
def load_texts(folder_path):
    texts = {}
    for file in os.listdir(folder_path):
        if file.lower().endswith('.txt'):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                texts[file] = f.read()
    return texts

# --- Hybrid Chunking ---
def chunk_text(text):
    chunks = []
    year_matches = list(re.finditer(r'\b(1[0-9]{3}|20[0-9]{2})\b', text))
    
    # Date-based chunking if years found
    if year_matches:
        current_chunk = []
        current_year = DEFAULT_YEAR
        pos = 0
        
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
    else:
        # Fixed-size chunking with overlap for texts without years
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + FIXED_CHUNK_SIZE, text_length)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "year": DEFAULT_YEAR,
                    "start": start,
                    "end": end
                })
            start += (FIXED_CHUNK_SIZE - CHUNK_OVERLAP)
    
    return chunks

def create_chunk(chunks, year, start, end):
    return {
        "text": "".join(chunks).strip(),
        "year": year,
        "start": start,
        "end": end
    }

# --- Document Processing ---
def process_documents(folder_path):
    texts = load_texts(folder_path)
    all_docs = []
    
    for filename, content in texts.items():
        chunks = chunk_text(content)
        for chunk in chunks:
            all_docs.append(Document(
                page_content=chunk["text"],
                metadata={
                    "source": filename,
                    "year": chunk["year"]
                }
            ))
    
    return all_docs

# --- Query Handling ---
def parse_query(query):
    doc = nlp(query)
    years = [int(y) for y in re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', query)]
    
    return {
        "years": years,
        "temporal": {
            "before": "before" in query.lower(),
            "after": "after" in query.lower(),
            "during": "during" in query.lower()
        }
    }

def build_filters(parsed):
    if not parsed["years"]:
        return None  # No filters for semantic search
    
    year = parsed["years"][0]
    filters = {"year": year}  # Default exact match
    
    if parsed["temporal"]["before"]:
        filters["year"] = {"$lt": year}
    elif parsed["temporal"]["after"]:
        filters["year"] = {"$gt": year}
    
    return filters

def retrieve(query, vectorstore):
    parsed = parse_query(query)
    filters = build_filters(parsed)
    
    # Use filters only if year is present
    if filters:
        try:
            return vectorstore.similarity_search(query, k=TOP_K, filter=filters)
        except Exception as e:
            print(f"Filter error: {e}, falling back to unfiltered search")
    
    # Default semantic search
    return vectorstore.similarity_search(query, k=TOP_K)

# --- Main Execution ---
if __name__ == "__main__":
    DATA_DIR = "/home/akash/HistoriChat/data"
    PERSIST_DIR = os.path.expanduser("~/Historichat_v2")  # New persist directory
    
    # Always process documents if data dir changes (for demo purposes)
    print("Processing documents...")
    docs = process_documents(DATA_DIR)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=PERSIST_DIR
    )
    
    # Test queries
    test_queries = [
        ("Date-based", "Events of 1066"),
        ("Date-range", "What happened before 1100 AD"),
        ("Entity-only", "Tell me about Alfred's childhood"),
        ("Mixed", "Norman conquest after 1000 AD"),
        ("No-year", "Boudica's leadership qualities")
    ]
    
    for header, query in test_queries:
        print(f"\n=== {header} Query: {query} ===")
        results = retrieve(query, vectorstore)
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {doc.metadata['source']}")
            print(f"Year: {doc.metadata.get('year', 'N/A')}")
            print(f"Text: {doc.page_content[:250]}...")