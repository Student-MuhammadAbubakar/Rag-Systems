# ============================================================
# FILE     : Downloadpage.py
# PROJECT  : Customer Support Chatbot
# PURPOSE  : Download customer support dataset from
#            Hugging Face and save as text files
# ============================================================

import os

# FIX: Set cache directory to a path with NO spaces
# Must be set BEFORE importing datasets
os.environ["HF_HOME"] = "C:\\huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "C:\\huggingface_cache"
# This tells Hugging Face to use C:\huggingface_cache
# instead of C:\Users\HS LAPTOP\.cache\huggingface
# The space in "HS LAPTOP" was causing the crash

from datasets import load_dataset #type:ignore
# Now imports AFTER setting the cache path

SAVE_DIR = "shopify-support"
os.makedirs(SAVE_DIR, exist_ok=True)
# Create the folder where text files will be saved

print("=" * 55)
print("  Downloading Customer Support Dataset")
print("  Source: Hugging Face (free, no scraping)")
print("=" * 55)

# Download the dataset
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)
print(f"  Total records: {len(dataset)}")
print(f"  Columns: {dataset.column_names}")
print()

# Save each record as a text file
saved = 0
skipped = 0

for i, record in enumerate(dataset):

    intent   = record.get("intent", "general")
    question = record.get("instruction", "").strip()
    answer   = record.get("response", "").strip()
    category = record.get("category", "").strip()

    if not question or not answer:
        skipped += 1
        continue
    # Skip any empty records

    # Build clean readable text content
    content = (
        f"Category: {category}\n"
        f"Intent: {intent}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n"
    )

    # Save as .txt file
    filename = f"record_{i:05d}_{intent}.txt"
    filepath = os.path.join(SAVE_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    saved += 1

    if saved % 2000 == 0:
        print(f"  Saved {saved} records so far...")

print()
print("=" * 55)
print("  Download complete!")
print(f"  Records saved   : {saved}")
print(f"  Records skipped : {skipped}")
print(f"  Files saved in  : {SAVE_DIR}/")
print("=" * 55)
print()
print("  Next step: python upload_vectors.py")