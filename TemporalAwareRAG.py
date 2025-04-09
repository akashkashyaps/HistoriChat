import os
import re
import shutil
import spacy
from typing import Dict, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# Initialize Spacy NLP
nlp = spacy.load("en_core_web_sm")

# --- Configuration ---
DEFAULT_YEAR = 0
TOP_K = 10
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DATA_DIR = "/home/akash/HistoriChat/data"
PERSIST_DIR = os.path.join(os.path.expanduser("~"), "Historichat")

# RAG Template
rag_template = ("""
You are "HistoriChat," an educational AI designed to teach users about historical figures through accurate information presentation.

## CORE FUNCTIONALITY
You present information about historical figures based solely on the historical context provided to you. You are not roleplaying as these characters but rather serving as a knowledgeable educator about them.

## ACCURACY PROTOCOL
1. Base ALL responses EXCLUSIVELY on the provided historical information in the context.
2. If asked about something not covered in the context:
   - Respond with: "I don't have that specific information about this historical figure. Would you like to know about something else related to them?"
   - DO NOT invent facts or speculate beyond the provided context.

3. When encountering conflicting historical accounts in the context:
   - Explicitly state: "Historical accounts differ on this matter..."
   - Present the different viewpoints: "Some sources suggest... while others claim..."
   - If appropriate, indicate which view has stronger historical support according to the context.

## INTERACTION STYLE
- Present information clearly and educationally.
- Focus on accuracy and educational value.
- Avoid first-person roleplaying as historical figures.
- Keep responses concise but informative.
- Reference only information contained in the provided context.

Remember: Your purpose is to educate about historical figures based solely on the provided context. Never fabricate information or answer questions that cannot be addressed with the provided context.

CONTEXT: {context}
QUESTION: {question}
HistoriChat's answer:
""")

prompt = PromptTemplate(template=rag_template, input_variables=["context", "question"])

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
            filtered_results = vectorstore.similarity_search(query, k=5, filter=filters)
            results.extend([(doc, "entity") for doc in filtered_results])
    except Exception as e:
        print(f"Entity search error: {e}")

    # Semantic search
    try:
        semantic_results = vectorstore.similarity_search(query, k=5)
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

# --- RAG Response Generation ---
def get_rag_response(query: str, llm, vectorstore, cross_encoder) -> Dict[str, Any]:
    # Retrieve and rerank
    results = retrieve(query, vectorstore)
    
    # Rerank with Cross-Encoder
    pairs = [(query, doc.page_content) for doc, _ in results]
    scores = cross_encoder.predict(pairs)
    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    top4 = scored_results[:4]

    # Prepare context
    context = "\n\n".join([doc.page_content for doc, _ in top4])
    
    # Generate response
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    
    return {
        "answer": answer,
        "context": context,
        "source_docs": [doc.metadata["source"] for doc, _ in top4]
    }

# --- Execution Pipeline ---
print("\n=== Path Verification ===")
print(f"Data directory exists: {os.path.exists(DATA_DIR)}")

# Delete existing vectorstore
if os.path.exists(PERSIST_DIR):
    print("Deleting old vectorstore...")
    shutil.rmtree(PERSIST_DIR)

print("\n=== Processing Documents ===")
docs = process_documents(DATA_DIR)
print(f"Total chunks created: {len(docs)}")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory=PERSIST_DIR,
    collection_name="HC-2"
)
print("Vectorstore created and saved.")

# Initialize models
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
llm = ChatOllama(model="llama3.1", temperature=0.2, frequency_penalty=0.5)

# Test queries
test_queries = [
    "tell me about alfred the great",
    "what happened before 1066",
    "what happened after 1066",
    "battle of hastings"
]

for query in test_queries:
    print("\n" + "="*40)
    print(f"Processing query: {query}")
    
    response = get_rag_response(query, llm, vectorstore, cross_encoder)
    
    print("\n=== Final Answer ===")
    print(response["answer"])
    
    print("\n=== Supporting Context ===")
    print(response["context"][:500] + "...")  # Show first 500 chars of context
    
    print("\n=== Source Documents ===")
    print(", ".join(set(response["source_docs"])))