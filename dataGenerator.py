from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

# Initialize Ollama components
ollama_llm = ChatOllama(model="llama3.1", temperature=0.2)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text") 

# Wrap components for RAGAS compatibility
generator_llm = LangchainLLMWrapper(ollama_llm)

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
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=ollama_embeddings
)

# Generate test dataset
testset_size = 1000
dataset = generator.generate_with_langchain_docs(split_docs, testset_size=testset_size, raise_exceptions=False)

# Save results
dataset.to_pandas().to_csv("testset_v1.csv", index=False)