import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True}
)


docs = []
with open("data/merged.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

# Convert to doc object
documents = []

for d in docs:
    metadata = {}

    for key in [
        "id", "source", "url", "name", "sku",
        "category", "subcategory",
        "price_current_kzt", "currency",
        "stock_almaty", "stock_astana"
    ]:
        if key in d and d[key] is not None:
            metadata[key] = d[key]

    documents.append(
        Document(
            page_content=d["text"],
            metadata=metadata
        )
    )

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunked_docs = text_splitter.split_documents(documents)

print(f"Original docs: {len(documents)}")
print(f"Chunked docs: {len(chunked_docs)}")


# Creating vector DB
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


print("Vector DB created with chunking and saved to ./chroma_db")