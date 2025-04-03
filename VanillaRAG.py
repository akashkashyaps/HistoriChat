from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
import torch
import time
from typing import List, Dict, Any
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the models to choose from
models = ["llama3.1:8b-instruct-q4_0", "qwen2.5:7b-instruct-q4_0", "gemma2:9b-instruct-q4_0", "phi3.5:3.8b-mini-instruct-q4_0", "mistral:7b-instruct-q4_0", "lly/InternLM3-8B-Instruct:8b-instruct-q4_0"]

# Load two documents
loader1 = TextLoader("Alfred_the_Great.txt")
loader2 = TextLoader("Boudica.txt") 
loader3 = TextLoader("William_the_Conqueror.txt") 

# Load the documents
loaded_documents1 = loader1.load()
loaded_documents2 = loader2.load()

# Combine the loaded documents
loaded_documents = loaded_documents1 + loaded_documents2

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1900, chunk_overlap=128) 

# Split the loaded documents into chunks
recreated_splits = text_splitter.split_documents(loaded_documents)

# Initialize Chroma vector store
import os
home_directory = os.path.expanduser("~")
persist_directory = os.path.join(home_directory, "HSv2", "vecdb")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"), collection_name="ROBIN-6")

# Initialize retrievers
retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# RAG template
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

# Function to get RAG response
def get_rag_response(query: str, llm) -> Dict[str, Any]:
    # Retrieve context from vector store
    context = retriever_vanilla.invoke(query)
    
    # Generate a response using the RAG pipeline
    result = (prompt | llm | StrOutputParser()).invoke({"question": query, "context": context})
    
    # Return both the answer and the retrieved context
    return {
        "answer": result,
        "context": context
    }

# Create test set
test = pd.read_csv('abcd.csv')
questions = test['question'].tolist()
ground_truths = test['ground_truth'].tolist()

# Loop through each model
for model in models:
    print(f"Running model: {model}")
    
    # Initialize the LLM
    llm = ChatOllama(model=model, temperature=0.2, frequency_penalty=0.5)
    
    # Create empty lists to store the results, context, and the time taken
    results = []
    contexts = []
    chain_time_list = []

    # Loop through each question
    for question in tqdm(questions):
        # Time the chain run process
        start_chain = time.time()
        rag_response = get_rag_response(question, llm)
        end_chain = time.time()
        
        chain_time = end_chain - start_chain
        chain_time_list.append(chain_time)  # Store chain run time
        
        # Store the answer and context
        results.append(rag_response["answer"])
        contexts.append(rag_response["context"])

    # Create a pandas DataFrame to store the results, context, and times taken
    df = pd.DataFrame({
        "Question": questions,
        "Answer": results,
        "Context": contexts,
        "Ground_Truth": ground_truths,
        "Chain_Time": chain_time_list
    })
    
    # Save the results to a CSV file
    df.to_csv(f'Output_{model.replace("/", "_")}.csv', index=False)