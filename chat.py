from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Embed chunks
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name= "learning_rag",
    embedding=embeddings

)


# Take user Input
user_query = input("Ask something : ")

# Relevant chunks from vector db
search_result = vector_db.similarity_search(query=user_query)

SYSTEM_PROMPT = f"""
You are a helful AI Assistant who answers user query based on available context retrived from a PDF file alog with page_contents and page_number
""" 