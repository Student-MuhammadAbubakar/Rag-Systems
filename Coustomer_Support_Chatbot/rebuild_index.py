import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

txt_files = glob.glob("shopify-support/*.txt")[:5000]
print(f"Using {len(txt_files)} files")

documents = []
for i, file_path in enumerate(txt_files):
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
    except:
        pass
    if (i+1) % 1000 == 0:
        print(f"Loaded {i+1} files...")

print(f"Total documents: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

BATCH = 500
db = None
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i+BATCH]
    print(f"Embedding batch {i//BATCH + 1}/{len(chunks)//BATCH + 1}...")
    if db is None:
        db = FAISS.from_documents(batch, embeddings)
    else:
        db.merge_from(FAISS.from_documents(batch, embeddings))

if db is not None:
    os.makedirs("faiss_index", exist_ok=True)
    db.save_local("faiss_index", index_name="cs_support")
print("Done!")
