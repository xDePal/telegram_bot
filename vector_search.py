from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cuda"}
)

# Load existing DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Query
query = "какие есть вакансии?"

results = vectorstore.similarity_search(
    query,
    k=5
)

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
