import pickle
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Load the embeddings database
with open('db_embeddings.pkl', 'rb') as f:
    db = pickle.load(f)

# Initialize OllamaEmbeddings
embeddings = OllamaEmbeddings(model="mistral")

# Function to query the database and process results
def query_database(question: str, top_k: int = 5):
    # Embed the question
    question_embedding = embeddings.embed_query(question)

    # Query the database using similarity_search
    similar_docs = db.similarity_search(question_embedding, k=top_k)

    # Process and print results
    for i, doc in enumerate(similar_docs):
        print(f"Result {i+1}:")
        # Implement summarization or key sentence extraction here
        print(doc.page_content)  # Placeholder for processed content
        print("-----------")

# Example usage
questions = [
    "Who are the authors of the paper?",
    "What is the title of the paper?",
    # ... other questions
]

# Process each question
for question in questions:
    print(f"Query: {question}")
    query_database(question)
    print("\n")
