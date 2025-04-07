from langchain_community.document_loaders import DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator


import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Initialize Ollama components
ollama_llm = ChatOllama(model="qwen2.5:7b-instruct-q4_0", temperature=0.2)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text") 


# Load documents
loader = DirectoryLoader("/home/akash/HistoriChat/data", glob="*.txt")

docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128
)
split_docs = text_splitter.split_documents(docs)

# Initialize test generator with Ollama components
generator = TestsetGenerator.from_langchain(ollama_llm, ollama_llm, ollama_embeddings)

# Generate test dataset
testset_size = 1000
dataset = generator.generate_with_langchain_docs(split_docs, testset_size=testset_size)

# Save results
dataset.to_pandas().to_csv("testset_v1.csv", index=False)