from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# Load existing DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Query
query = "чем занимается эта компания?"

results = vectorstore.similarity_search(
    query,
    k=3
)

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print(doc.metadata)