from langchain_community.document_loaders import DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

import os
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1/chat/completions"

# Initialize Ollama components
base_llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    temperature=0.2,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_BASE_URL"]
)

llm = LangchainLLMWrapper(base_llm)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text") 
embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)


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
generator = TestsetGenerator(llm = llm, embedding_model = embeddings)

# Generate test dataset
testset_size = 10
dataset = generator.generate_with_langchain_docs(split_docs, testset_size=testset_size, raise_exceptions=False)

# Save results
dataset.to_pandas().to_csv("testset_v1.csv", index=False)