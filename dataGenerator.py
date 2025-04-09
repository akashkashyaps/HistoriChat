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

# Define batch size
batch_size = 100  # Number of samples to process per batch
testset_size = 1000  # Total number of samples to generate

# Calculate the number of batches
num_batches = (testset_size + batch_size - 1) // batch_size  # Ceiling division

# Initialize an empty list to store results
all_results = []

# Process each batch
for batch_idx in range(num_batches):
    print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches}...")
    
    # Calculate start and end indices for the current batch
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, testset_size)
    
    try:
        # Generate the current batch
        batch_dataset = generator.generate_with_langchain_docs(
            recreated_splits, 
            testset_size=(end_idx - start_idx),  # Size of the current batch
            raise_exceptions=False  # Skip errors within the batch
        )
        
        # Append the batch results to the main list
        all_results.append(batch_dataset)
        print(f"[INFO] Successfully processed batch {batch_idx + 1}/{num_batches}.")
    
    except Exception as e:
        # Log the error and continue with the next batch
        print(f"[ERROR] Failed to process batch {batch_idx + 1}/{num_batches}: {e}")
        continue

# Combine all batch results into a single dataset
if all_results:
    final_dataset = all_results[0].concat(all_results[1:]) if len(all_results) > 1 else all_results[0]
    final_dataset.to_pandas().to_csv("testset_v2.csv", index=False)
    print("[INFO] Dataset saved successfully.")
else:
    print("[INFO] No data generated. Dataset is empty.")




