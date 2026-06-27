# RAG Systems

This repository contains a Retrieval-Augmented Generation (RAG) system for a customer support assistant. The project combines a vector database, embedding models, and a large language model to answer support questions using a custom knowledge base instead of relying only on the model's pretrained knowledge.

## Overview

The system works in three main stages:

1. Load support knowledge from text files.
2. Convert those documents into embeddings and store them in a FAISS vector index.
3. Retrieve the most relevant chunks for a user question and generate a grounded answer with Groq.

This makes the chatbot more useful for domain-specific questions such as order cancellation, refunds, payments, shipping, and account issues.

## Project Structure

- [Coustomer_Support_Chatbot](Coustomer_Support_Chatbot) — Streamlit chatbot app and indexing scripts
- [Coustomer_Support_Chatbot/shopify-support](Coustomer_Support_Chatbot/shopify-support) — support documents used as the knowledge base
- [Coustomer_Support_Chatbot/faiss_index](Coustomer_Support_Chatbot/faiss_index) — FAISS vector store generated from the documents

## Technologies Used

- Python
- LangChain
- Hugging Face embeddings
- FAISS vector search
- Groq LLM
- Streamlit
- Python-dotenv

## Prerequisites

Before running the project, make sure you have:

- Python 3.9 or newer
- pip installed
- A Groq API key
- Internet access for downloading embedding and LLM models

## Installation

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Build the Knowledge Base

The chatbot uses text files from the support knowledge folder. To create the FAISS index, run:

```bash
python Coustomer_Support_Chatbot/Chatbot_Upload.py
```

This script will:

- load all `.txt` files from the support folder
- split them into smaller chunks
- generate embeddings
- build and save the FAISS index

> The first index build may take several minutes depending on the number of documents.

## Run the Chatbot UI

Start the Streamlit app:

```bash
streamlit run Coustomer_Support_Chatbot/Chatbot_ui.py
```

Then open the local URL shown in the terminal.

## How the RAG Pipeline Works

1. A user asks a question in the Streamlit interface.
2. The app rewrites the question using chat history when needed.
3. Relevant documents are retrieved from the FAISS index.
4. The retrieved context is passed to the LLM along with the user question.
5. The model generates a helpful support-style response grounded in the retrieved content.

## Recommended Usage

You can use the chatbot for questions like:

- How do I cancel my order?
- Where is my refund?
- Why was my payment declined?
- How do I return a product?
- How do I reset my password?

## Notes

- If you add or change support documents, rebuild the FAISS index.
- If the FAISS index is missing or outdated, the chatbot may return weak or incomplete answers.
- The app is designed for customer support use cases and can be extended to other domains.

## Future Improvements

Possible improvements include:

- adding a web interface for uploading new documents
- switching to a more advanced embedding model
- adding authentication for production use
- logging and analytics for chatbot usage
