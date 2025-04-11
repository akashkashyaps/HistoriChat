from langchain_community.document_loaders import DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import json
import signal

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Initialize Ollama components
ollama_llm = ChatOllama(model="qwen2.5:7b-instruct-q4_0", temperature=0.2)
llm = LangchainLLMWrapper(ollama_llm)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text") 
embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

# Load two documents
loader1 = TextLoader("/home/akash/HistoriChat/data/Alfred_the_Great.txt")
loader2 = TextLoader("/home/akash/HistoriChat/data/Boudica.txt") 
loader3 = TextLoader("/home/akash/HistoriChat/data/William_the_Conqueror.txt") 

# Load the documents
loaded_documents1 = loader1.load()
loaded_documents2 = loader2.load()
loaded_documents3 = loader3.load()

# Combine the loaded documents
loaded_documents = loaded_documents1 + loaded_documents2 + loaded_documents3

# ✅ Confirm document loading
print(f"[INFO] Loaded {len(loaded_documents)} documents.")
total_chars = sum(len(doc.page_content) for doc in loaded_documents)
print(f"[INFO] Total characters loaded: {total_chars}")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1900, chunk_overlap=128) 

# Split the loaded documents into chunks
recreated_splits = text_splitter.split_documents(loaded_documents)

# Initialize test generator with Ollama components
generator = TestsetGenerator(llm = llm, embedding_model= embeddings)

# Generation parameters
TARGET_SAMPLES = 1000
BATCH_SIZE = 50  # Reduced for better error recovery
num_batches = (TARGET_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

# Initialize progress bar
pbar = tqdm(total=TARGET_SAMPLES, desc="Generating QnA Pairs", unit="sample")

# Output file configuration
OUTPUT_CSV = "testset_v2.csv"
header_written = False

try:
    for batch_idx in range(num_batches):
        current_target = min(BATCH_SIZE, TARGET_SAMPLES - (batch_idx * BATCH_SIZE))
        
        try:
            # Generate batch
            batch_dataset = generator.generate_with_langchain_docs(
                recreated_splits,
                testset_size=current_target,
                raise_exceptions=False
            )
            
            # Convert to DataFrame
            batch_df = batch_dataset.to_pandas()
            
            # Save immediately with progress update
            if not batch_df.empty:
                # Write header only once
                write_header = not os.path.exists(OUTPUT_CSV) or os.stat(OUTPUT_CSV).st_size == 0
                batch_df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
                
                # Update progress
                samples_generated = len(batch_df)
                pbar.update(samples_generated)
                
        except Exception as e:
            print(f"\n[ERROR] Batch {batch_idx+1} failed: {str(e)[:200]}")
            continue

finally:
    pbar.close()
    print(f"\n[COMPLETED] Final dataset saved to {OUTPUT_CSV}")

# Optional: Load and verify final dataset
if os.path.exists(OUTPUT_CSV):
    final_df = pd.read_csv(OUTPUT_CSV)
    print(f"\n[FINAL STATS] Total samples generated: {len(final_df)}")




