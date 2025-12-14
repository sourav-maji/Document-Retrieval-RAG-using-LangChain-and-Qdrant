from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()

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

context = "\n\n\n".join([ f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_result ])

SYSTEM_PROMPT = f"""
You are a helful AI Assistant who answers user query based on available context retrived from a PDF file alog with page_contents and page_number


You should only ans the user based on the following context and navigate the user to open the right number to know more

Context : 
{context}
""" 

response = openai_client.chat.completions.create(
    model="gpt-5",
    messages=[
        { "role" : "system", "content" : SYSTEM_PROMPT},
        { "role" : "user", "content" : user_query}

    ]
)

print(f"🤖: {response.choices[0].message.content}")