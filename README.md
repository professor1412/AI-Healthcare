# Healthcare AI Assistant

A FastAPI-based REST API that ingests medical PDF reports (cardiology, psychology, pulmonology) and answers clinical queries via a modular LangChain + Groq backend, supplemented by a trusted web-search tool (Tavily).

---

## 🌟 Features

- **Multi-disciplinary Retrieval**  
  – Embed and index uploaded PDF reports with HuggingFace MiniLM embeddings stored in ChromaDB  
  – Limit top-𝑘 (default 3) chunks per query for concise context  

- **Domain-specific Agents**  
  – Cardiologist, Psychologist, Pulmonologist “ReAct” agents built on Llama-3.3-70B via Groq  
  – Modular prompt templates per specialty  

- **Combined RAG Pipeline**  
  – Parallel RAG chain to produce a comprehensive “Multidisciplinary Team” report  
  – Output parsed to plain text via StrOutputParser  

- **Trusted Web Search**  
  – Built-in Tavily tool anchoring results to authoritative domains (Mayo Clinic, NIH, WHO, CDC, etc.)  
  – Advanced search depth, raw content, answer snippets  

- **CORS-enabled**  
  – Allows cross-origin calls from any frontend  

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/your-org/healthcare-ai-assistant.git
cd healthcare-ai-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
