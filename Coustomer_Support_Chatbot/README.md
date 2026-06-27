# Customer Support Chatbot

This folder contains a Retrieval-Augmented Generation (RAG) powered customer support chatbot. It uses a local knowledge base of support documents, retrieves the most relevant information, and generates helpful responses with a large language model.

## What This App Does

The chatbot can answer customer support questions such as:

- order cancellation
- refund status
- payment issues
- shipping questions
- returns and account-related concerns

It is designed to provide quick, context-aware answers based on your support documents rather than relying only on general model knowledge.

## Files in This Folder

- [Chatbot_ui.py](Chatbot_ui.py) — the Streamlit web app for chatting with the assistant
- [Chatbot_Upload.py](Chatbot_Upload.py) — builds the FAISS vector index from the support text files
- [downloadpages.py](downloadpages.py) — optional script for downloading support-related pages
- [shopify-support](shopify-support) — sample support documents used as the knowledge base
- [faiss_index](faiss_index) — generated vector index used by the chatbot

## Prerequisites

Make sure you have:

- Python 3.9+
- pip
- a Groq API key
- the dependencies from the repository requirements file

## Setup

1. Open the project root folder.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Build the Vector Database

Run the following command from the repository root:

```bash
python Coustomer_Support_Chatbot/Chatbot_Upload.py
```

This will:

- read all `.txt` files in the support folder
- split them into manageable chunks
- generate embeddings
- save a FAISS index for retrieval

## Run the Chatbot

Launch the app with:

```bash
streamlit run Coustomer_Support_Chatbot/Chatbot_ui.py
```

The interface will open in your browser, and you can start asking support questions.

## Example Questions

Try questions like:

- How do I cancel my order?
- Where is my refund?
- My payment was declined, what should I do?
- How do I return a product?
- How do I reset my password?

## Troubleshooting

- If responses seem inaccurate, rebuild the FAISS index after updating the knowledge files.
- If the app cannot start, confirm that your Groq API key is set correctly.
- If the index is missing, run the upload script again.

## Notes

The chatbot is best when the knowledge base contains clear, up-to-date support content. Updating the support documents and rebuilding the index will improve answer quality.
