# RAG_financial

PolicyPilot AI – A RAG-based Compliance Assistant
Project Name:
PolicyPilot AI – Your Compliance & Finance Copilot (Built using RAG + LLMs)

🚨 Problem
Employees at financial institutions like Bank of Ireland struggle to search through long and complex regulatory PDFs, leading to inefficiencies and compliance delays.

🎯 Objective
Build a Retrieval-Augmented Generation (RAG) system that lets users ask natural language questions and receive contextual answers from internal policy and financial documents.

📂 Data Used
PDFs such as:

Q1 2025 Interim Report

FY24 Earnings Call Transcript

Investor Presentations

Fitch Rating Memos, etc.

🧱 Architecture Components
PDF Parsing: PyPDF2 to extract text

Chunking: LangChain splits text

Embeddings: MiniLM generates vector embeddings

Vector DB: FAISS for similarity search

LLM: Zephyr or Mistral 7B generates responses

UI: Built with Streamlit for interaction

🔁 RAG Workflow
User enters a query

FAISS retrieves top 3 similar chunks

Prompt = context + question

Zephyr-7B answers using that context

Output + source PDF file is shown in the UI

🌟 Key Features
Lightning-fast answers from 100+ page PDFs

Reduces dependency on legal teams

Transparent, source-linked answers

Scalable across multiple finance and insurance firms

🔧 Future Plans
Add support for Word/Excel files

Enterprise-ready deployment on Azure/AWS

Role-based access & audit logging

Query analytics dashboard

Fine-tuning with internal data

✅ Evaluation Strategy
Manual Verification of accuracy

Consistency Checks via varied queries

Source File Display for every answer

Context-Limited Prompts to avoid hallucination

🔐 Hallucination Prevention
Top-3 Chunk retrieval

Overlapping chunks

Strict prompt formatting

Source metadata retention

Using lightweight, instruction-following LLMs
