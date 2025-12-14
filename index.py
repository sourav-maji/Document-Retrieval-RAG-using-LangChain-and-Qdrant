from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

load_dotenv()



# PDF path
pdf_path = Path(__file__).parent / "Mastering_HTTP_in Node.js_ From_Theory_to_Implementation.pdf"

# Load PDF
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print("First chunk content:", chunks[0].page_content)

# Embed chunks
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding= embeddings,
    url="http://localhost:6333",
    collection_name= "learning_rag"
)

print("Indexing of documents is done...")