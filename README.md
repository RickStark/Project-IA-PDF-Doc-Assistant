# 🤖 PDF Intelligence Chatbot: RAG-based Research Assistant

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to assist researchers and students in navigating large volumes of scientific articles. 

Instead of relying on the general knowledge of a Pre-trained Model, this chatbot uses **Vector Search** to retrieve relevant context from specific PDF documents, ensuring grounded and technically accurate responses.

## 🏗️ Technical Architecture
The system follows a modern AI engineering pipeline:

1.  **Ingestion:** Extracts raw text from PDF files using `PyPDF`.
2.  **Document Chunking:** Implements `RecursiveCharacterTextSplitter` to break down text into semantically meaningful segments while maintaining context.
3.  **Vector Embeddings:** Transforms text chunks into high-dimensional vectors using `Google Generative AI Embeddings`.
4.  **Vector Store:** Utilizes `ChromaDB` (or similar) as a local vector database for efficient similarity searches.
5.  **Retrieval & Generation:** Orchestrates the flow where the most relevant chunks are retrieved and fed into the `Gemini-Pro` model as grounding context.



## 🚀 Key Features
* **Contextual Grounding:** Minimizes hallucinations by forcing the model to answer based on provided PDF data.
* **Semantic Search:** Finds information based on meaning rather than just keyword matching.
* **Scalability:** Designed to handle multiple technical documents simultaneously.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Orchestration:** LangChain
* **AI Model:** Google Gemini Pro
* **Vector Database:** ChromaDB
* **Environment:** Docker / Python Virtual Environment

## 📈 Insights & Learning
* **Cost Efficiency:** Implementing RAG is significantly more cost-effective than Fine-Tuning for domain-specific knowledge retrieval.
* **Robustness:** By adjusting the `chunk_size` and `overlap` parameters, I optimized the balance between detailed retrieval and LLM context window limits.
* **Professional Application:** This prototype serves as a foundation for my Master's Thesis (TFM) project on **Auto-healing Data Pipelines**, where AI agents interpret system logs and technical documentation.

## 📁 Project Structure
```text
├── inputs/           # Source PDF documents
├── src/              # Python source code
├── requirements.txt  # Project dependencies
└── README.md         # Documentation