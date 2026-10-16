import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# FAISS has no official standalone LangChain package yet, so it still comes from
# langchain-community (which is deprecated but functional).
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY_DIR = os.path.join(BASE_DIR, "policy_docs")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store(policy_dir: str = DEFAULT_POLICY_DIR) -> FAISS:
    """Load all policy docs and build FAISS index."""
    docs = []
    for filename in sorted(os.listdir(policy_dir)):
        if filename.endswith('.txt'):
            path = os.path.join(policy_dir, filename)
            with open(path, encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": path}))

    if not docs:
        raise FileNotFoundError(f"No .txt policy documents found in {policy_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_policy_context(vectorstore: FAISS, query: str, k: int = 3) -> str:
    """Retrieve the top-k relevant policy chunks for a given signal query."""
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])
