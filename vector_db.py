import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True}
)


with open("data/output.md", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Wrap the whole text in a single Document — no metadata needed
documents = [Document(page_content=raw_text)]

print(f"Total characters: {len(raw_text)}")


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