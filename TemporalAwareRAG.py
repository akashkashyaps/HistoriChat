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

#  Load NLP Model for Entity Extraction
nlp = spacy.load("en_core_web_sm")

folder_path = "/home/akash/HistoriChat/data"

# Load and parse text
def load_text_files(folder_path):
    """Load all .txt files from a folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    texts = {}
    for file in files:
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            texts[file] = f.read()
    return texts

# Extract metadata
def extract_years(text):
    """Extracts years (1000-2099) from text using regex."""
    year_matches = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
    return sorted({int(year) for year in year_matches})

def extract_entities(text):
    """Extracts historical entities (PERSON, EVENT, DATE, GPE) using Spacy."""
    doc = nlp(text)
    entities = defaultdict(list)
    
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "EVENT", "DATE", "GPE"}: 
            entities[ent.label_].append(ent.text)
    
    # Deduplicate entities
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return dict(entities)

# Timeline-based Chunking
def chunk_by_timeline(text, chunk_size=1000, max_gap=20):
    """
    Splits text into timeline-based chunks.
    Chunks are formed by keeping sections with close years together.
    
    - chunk_size: max characters per chunk
    - max_gap: max allowed gap between years to remain in the same chunk
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    raw_chunks = splitter.split_text(text)
    
    enriched_chunks = []
    current_chunk = []
    current_years = []

    for chunk in raw_chunks:
        years = extract_years(chunk)
        
        if not current_chunk:  
            # Start first chunk
            current_chunk.append(chunk)
            current_years.extend(years)
            continue

        # If the new chunk’s years are close to the current chunk, merge them
        if years and current_years and (min(years) - max(current_years) <= max_gap):
            current_chunk.append(chunk)
            current_years.extend(years)
        else:
            # Save the previous chunk
            enriched_chunks.append({
                "text": " ".join(current_chunk),
                "years": sorted(set(current_years)),
                "start_year": min(current_years) if current_years else None,
                "end_year": max(current_years) if current_years else None
            })
            # Start a new chunk
            current_chunk = [chunk]
            current_years = years

    # Save last chunk
    if current_chunk:
        enriched_chunks.append({
            "text": " ".join(current_chunk),
            "years": sorted(set(current_years)),
            "start_year": min(current_years) if current_years else None,
            "end_year": max(current_years) if current_years else None
        })
    
    return enriched_chunks

# Create vectorstore
def process_and_store_in_chromadb(folder_path):
    """Loads, processes, and stores text documents in ChromaDB with metadata."""
    
    # Load text files
    texts = load_text_files(folder_path)
    all_documents = []

    # Process each text file
    for filename, text in texts.items():
        chunks = chunk_by_timeline(text)
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "file": filename,
                "chunk_index": i,
                "start_year": chunk["start_year"],
                "end_year": chunk["end_year"],
                "years": chunk["years"],
                "entities": extract_entities(chunk["text"])
            }
            all_documents.append({"text": chunk["text"], "metadata": metadata})

    # Setup ChromaDB storage with LangChain
    home_directory = os.path.expanduser("~")
    persist_directory = os.path.join(home_directory, "HistoriChat")
    
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"), 
        collection_name="HC-1"
    )

    # Insert documents into ChromaDB
    for idx, doc in enumerate(all_documents):
        vectorstore.add_texts(
            texts=[doc["text"]],
            metadatas=[doc["metadata"]]
        )

    print(f"Stored {len(all_documents)} documents in ChromaDB.")

    return vectorstore

# -------------------------------------------------------------------------------------------------------------------

# Extract Query Metadata (Entities, Dates, Timeline Keywords)

def parse_query(query):
    """
    Extracts key information from the query:
    - Entities (People, Events, Locations)
    - Years
    - Temporal keywords (before, after, during, first, second, etc.)
    """
    doc = nlp(query)
    entities = {"PERSON": [], "EVENT": [], "DATE": [], "GPE": []}
    
    # Extract Named Entities
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    # Extract Years (e.g., "1066")
    years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', query)
    years = sorted(set(map(int, years)))

    # Extract Temporal Keywords
    temporal_keywords = []
    if "before" in query.lower():
        temporal_keywords.append("before")
    if "after" in query.lower():
        temporal_keywords.append("after")
    if "during" in query.lower():
        temporal_keywords.append("during")
    if any(word in query.lower() for word in ["first", "second", "third"]):
        temporal_keywords.append("ordered")
    
    return {
        "entities": entities,
        "years": years,
        "temporal_keywords": temporal_keywords
    }

# Custom retreiver with filtering
def retrieve_historical_facts(query):
    """
    Retrieves relevant historical facts based on the query.
    - Filters results based on entities, years, and timeline keywords.
    - Uses ChromaDB with LangChain's retriever.
    - Sorts results chronologically if needed.
    """
    # Parse query metadata
    parsed_query = parse_query(query)
    search_filter = {}
    conditions = []

    # Apply filters based on extracted data
    if parsed_query["entities"]["PERSON"]:
        conditions.append({
            "entities.PERSON": {"$contains": parsed_query["entities"]["PERSON"][0]}
        })
    
    if parsed_query["entities"]["EVENT"]:
        conditions.append({
            "entities.EVENT": {"$contains": parsed_query["entities"]["EVENT"][0]}
        })

    # Handle temporal filters
    if parsed_query["years"]:
        year = parsed_query["years"][0]
        if "before" in parsed_query["temporal_keywords"]:
            conditions.append({"start_year": {"$lt": year}})
        elif "after" in parsed_query["temporal_keywords"]:
            conditions.append({"start_year": {"$gt": year}})
        elif "during" in parsed_query["temporal_keywords"]:
            conditions.append({
                "$and": [
                    {"start_year": {"$lte": year}},
                    {"end_year": {"$gte": year}}
                ]
            })

    # Combine conditions with logical AND
    if conditions:
        if len(conditions) == 1:
            search_filter = conditions[0]
        else:
            search_filter = {"$and": conditions}

    # Load ChromaDB
    home_directory = os.path.expanduser("~")
    persist_directory = os.path.join(home_directory, "HistoriChat")
    
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"), 
        collection_name="HC-1"
    )
    
    # Use LangChain Retriever with constructed filter
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 10,
            "filter": search_filter if search_filter else None
        }
    )
    
    results = retriever.invoke(query)

    # Sort results chronologically if requested
    if "ordered" in parsed_query["temporal_keywords"]:
        results = sorted(
            results, 
            key=lambda doc: doc.metadata.get("start_year", 9999)
        )

    return results

# Example usage
query = "What did William the Conqueror do before 1066?"
retrieved_docs = retrieve_historical_facts(query)

print(f"Results for query: '{query}'\n")
for doc in retrieved_docs:
    print(f"Year Range: {doc.metadata.get('start_year')} - {doc.metadata.get('end_year')}")
    print(f"Text: {doc.page_content[:300]}...\n")
