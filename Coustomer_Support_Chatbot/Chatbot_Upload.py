from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import glob
import os
def uploadTXT():
    txt_files = glob.glob("shopify-support/*.txt")
    txt_files = list(set(txt_files))
    print(f"Found {len(txt_files)} TXT files")
    document = []
    for i, file_path in enumerate(txt_files):
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            document.extend(loader.load())
        except Exception as e:
            print(f"  [SKIP] {file_path} → {e}")

        if (i + 1) % 5000 == 0:
            print(f"  Loaded {i + 1}/{len(txt_files)} files...")
    print(f"\n{len(document)} files loaded successfully")
    if len(document) == 0:
        print("ERROR: No documents loaded!")
        return
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    split_documents = splitter.split_documents(documents=document)
    print(f"{len(split_documents)} chunks created")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    BATCH_SIZE = 500
    print(f"\nEmbedding {len(split_documents)} chunks")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {len(split_documents) // BATCH_SIZE + 1}")
    print("This will take 5-10 minutes, please wait...\n")

    db = None
    for i in range(0, len(split_documents), BATCH_SIZE):

        batch = split_documents[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = len(split_documents) // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} "
              f"({i} → {min(i + BATCH_SIZE, len(split_documents))} chunks)...")
        if db is None:
            db = FAISS.from_documents(
                documents=batch,
                embedding=embeddings
            )
        else:
            batch_db = FAISS.from_documents(
                documents=batch,
                embedding=embeddings
            )
            db.merge_from(batch_db)
    os.makedirs("faiss_index", exist_ok=True)
    db.save_local(folder_path="faiss_index", index_name="cs_support")
    print("\nFAISS index saved successfully!")
    print("Files created:")
    print("  faiss_index/cs_support.faiss")
    print("  faiss_index/cs_support.pkl")
def faiss_query():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local(
        folder_path="faiss_index",
        index_name="cs_support",
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    query = "How do I cancel my order?"
    docs = new_db.similarity_search(query)
    for doc in docs:
        print("##----file----##")
        print(doc.metadata['source'])
        print("##----content----##")
        print(doc.page_content)
        print()
if __name__ == "__main__":
    uploadTXT()
    faiss_query()