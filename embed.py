import os
import textract
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Import the Document class
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

# Convert PDF to text
pdf_path = "./dataset/file.pdf"  # Replace with your file path
doc = textract.process(pdf_path)

# Convert the binary content to string
text = doc.decode('utf-8')

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Function to count tokens
def count_tokens(text):
    return len(tokenizer.encode(text))

# Define separators for splitting text
separators = ['\n', '.', '!', '?']  # Add more separators as per your document's structure

# Initialize RecursiveCharacterTextSplitter with defined separators
text_splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=512, chunk_overlap=24, length_function=count_tokens)
chunks = text_splitter.split_text(text)

# Convert chunks into Document objects
documents = [Document(page_content=chunk) for chunk in chunks]

# Initialize Ollama Embeddings
embeddings = OllamaEmbeddings(model="mistral")

# Create vector database using the Document objects
db = FAISS.from_documents(documents, embeddings)

# Save the embeddings
embeddings_file = 'db_embeddings.pkl'
with open(embeddings_file, 'wb') as f:
    pickle.dump(db, f)

print(f"Embeddings saved to {embeddings_file}")
