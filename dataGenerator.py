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

# Generate test dataset with error handling
testset_size = 1000

try:
    dataset = generator.generate_with_langchain_docs(
        recreated_splits, 
        testset_size=testset_size, 
        raise_exceptions=False  # This ensures exceptions are not raised
    )
except KeyError as e:
    print(f"[ERROR] KeyError encountered: {e}. Skipping problematic entry.")
    # Optionally, log the error or handle it in a specific way
except TypeError as e:
    print(f"[ERROR] TypeError encountered: {e}. Skipping problematic entry.")
    # Optionally, log the error or handle it in a specific way
except Exception as e:
    print(f"[ERROR] Unexpected error encountered: {e}. Skipping problematic entry.")
    # Optionally, log the error or handle it in a specific way

# Save results if dataset generation was successful
if 'dataset' in locals():
    dataset.to_pandas().to_csv("testset_v2.csv", index=False)
    print("[INFO] Dataset saved successfully.")
else:
    print("[INFO] Dataset generation failed. No data to save.")



