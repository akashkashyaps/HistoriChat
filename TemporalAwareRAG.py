import os
import re
import spacy
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize Spacy for query parsing only
nlp = spacy.load("en_core_web_sm")

# --- Configuration ---
MAX_CHUNK_GAP = 1000
DEFAULT_YEAR = 0
TOP_K = 5

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

# --- Document Processing ---
def process_documents(folder_path):
    texts = load_texts(folder_path)
    all_docs = []
    
    for filename, content in texts.items():
        chunks = chunk_by_dates(content)
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
    filters = {}
    if parsed["years"]:
        year = parsed["years"][0]
        if parsed["temporal"]["before"]:
            filters["year"] = {"$lt": year}
        elif parsed["temporal"]["after"]:
            filters["year"] = {"$gt": year}
        else:
            filters["year"] = year
    return filters

def retrieve(query, vectorstore):
    parsed = parse_query(query)
    filters = build_filters(parsed)
    
    if filters:
        try:
            return vectorstore.similarity_search(query, k=TOP_K, filter=filters)
        except:
            return vectorstore.similarity_search(query, k=TOP_K)
    
    return vectorstore.similarity_search(query, k=TOP_K)

# --- Main Execution ---
if __name__ == "__main__":
    DATA_DIR = "/home/akash/HistoriChat/data"
    PERSIST_DIR = os.path.expanduser("~/Historichat")
    
    if not os.path.exists(PERSIST_DIR):
        docs = process_documents(DATA_DIR)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=PERSIST_DIR,
            collection_name="HC-3"
        )
    else:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )
    
    test_queries = [
        ("Date-based", "Events of 1066"),
        ("Date-range", "What happened before 1100 AD"),
        ("Entity-only", "Tell me about Alfred's childhood"),
        ("Mixed", "Norman conquest after 1000 AD")
    ]
    
    for header, query in test_queries:
        print(f"\n=== {header} Query: {query} ===")
        results = retrieve(query, vectorstore)
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {doc.metadata['source']}")
            print(f"Year: {doc.metadata.get('year', 'N/A')}")
            print(f"Text: {doc.page_content[:250]}...")