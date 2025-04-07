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

# Combine the loaded documents
loaded_documents = loaded_documents1 + loaded_documents2

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1900, chunk_overlap=128) 

# Split the loaded documents into chunks
recreated_splits = text_splitter.split_documents(loaded_documents)

# Initialize test generator with Ollama components
generator = TestsetGenerator(llm = llm, embedding_model= embeddings)

# Generate test dataset
testset_size = 10
dataset = generator.generate_with_langchain_docs(recreated_splits, testset_size=testset_size, raise_exceptions=False)

# Save results
dataset.to_pandas().to_csv("testset_v1.csv", index=False)