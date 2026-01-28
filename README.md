## RAG Port Project – Unified RAG / LangGraph Backend

This project is a **RAG (Retrieval-Augmented Generation)** system for ports, built around:

- **FastAPI** backend (`RAG-LLM-for-Ports-backend`)
- **LangChain + ChromaDB** for retrieval
- **LangGraph** for workflow-style recommendations and decision gates
- Optional **DB setup** and **multilingual/translation** helpers in sibling folders

The system is designed **never to return generic or “default” answers**. If it cannot answer from the indexed data or a workflow fails, it returns a **clear error** instead of fabricating a response.

---

## 1. Project Structure

- **Root**
  - `README.md` – This file
  - `.env` – Main environment file (created with `create_env.sh`)
  - `create_env.sh` – Helper to create `.env`
  - `test_suite.py` – Comprehensive test runner for the whole project
- **RAG-LLM-for-Ports-backend/**
  - `src/` – Main backend code
    - `api/server.py` – FastAPI app entrypoint
    - `port_information_retrieval.py` – Core RAG engine (Chroma + OpenAI)
    - `graph/port_graph.py` – LangGraph workflow
    - `routers/` – Intent routing & decision gates
    - `utils/` – Multilingual, loaders, schemas, etc.
  - `requirements.txt` – Backend Python dependencies
  - `setup_unified.py` – Unified data ingestion / indexing script
  - `START_SERVER.sh` – Helper to start the FastAPI server
  - `test_comprehensive.py` – Backend-level comprehensive tests
- **RAG-LLM-for-Ports-backend_dbsetup/**
  - DB-oriented setup scripts + `test_comprehensive.py`
- **RAG-LLM-for-Ports-backend_translate/**
  - Multilingual / translation helpers + `test_comprehensive.py`


---

## 2. Prerequisites

- **Python**: 3.10+ (project was tested with Python 3.11)
- **pip** and optionally **virtualenv** or `python -m venv`
- macOS or Linux (scripts are Bash-based; Windows users can use WSL)

---

## 3. Environment Setup

From the project root:




### 3.1 Configure `.env`

Make sure env is set up

## 4. Indexing Data (Optional but Recommended)

To make the RAG system useful, you need documents indexed into ChromaDB.

From the project root:

```bash
cd RAG-LLM-for-Ports-backend

# Run the unified setup script (adjust data paths inside the script if needed)
python setup_unified.py
```

If you want more granular control, you can also use the scripts under:

- `RAG-LLM-for-Ports-backend_dbsetup/` – CSV / DB-oriented setup
- `RAG-LLM-for-Ports-backend_translate/` – Multilingual & OCR utilities

These folders have their own `requirements.txt` and `test_comprehensive.py`, but they are not required for a basic backend run.

---

## 5. Running the FastAPI Server

From the project root:

```bash
cd RAG-LLM-for-Ports-backend
chmod +x START_SERVER.sh
./START_SERVER.sh
```

This will:

- Ensure it runs from the backend folder
- Load environment variables (including `OPENAI_API_KEY`)
- Start the FastAPI app on `http://0.0.0.0:8000` (configurable via `.env`)

You can visit:

- `http://localhost:8000/health` – Simple health check
- `http://localhost:8000/docs` – Swagger UI with interactive API docs

---

## 6. Core Endpoints

### 6.1 `/ask` – Direct RAG Question Answering

- **Method**: `POST /ask`
- **Body**:

```json
{
  "query": "What are the safety protocols for vessel berthing?"
}
```

- **Behavior**:
  - Uses `PortInformationRetrieval` to:
    - Embed the query
    - Retrieve relevant docs from Chroma
    - Call an OpenAI chat model to generate an answer
  - Returns a grounded `answer` and `sources`
  - If no relevant docs or empty answer, it raises a **clear error** (no default/fabricated answer).

Example (from another terminal, with the server running):

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the safety protocols for vessel berthing?"}'
```

### 6.2 `/ask_graph` – LangGraph Workflow

- **Method**: `POST /ask_graph`
- **Body**:

```json
{
  "query": "What are the safety protocols for vessel berthing?"
}
```

- **Behavior**:
  - Runs a full **LangGraph workflow**:
    - Intent analysis
    - RAG-based retrieval
    - Parallel scenario analyses
    - Freshness / confidence / compliance decision gates
    - Final recommendation + alternative scenarios + sources
  - Returns:
    - `recommendation`
    - `alternative_scenarios`
    - `sources`
    - `decision_gates` status
  - If the workflow cannot produce a valid recommendation, the server returns a **4xx/5xx error** with details instead of a default answer.

Example:

```bash
curl -X POST "http://localhost:8000/ask_graph" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the safety protocols for vessel berthing?"}'
```

---

## 7. Running Tests

### 7.1 Main Project Test Suite

From the project root:


This suite:

- Verifies environment setup and `.env`
- Checks backend imports
- Exercises core RAG (`PortInformationRetrieval`)
- Exercises the LangGraph workflow
- Optionally tests API endpoints (requires the server to be running)
- Confirms project file structure
- Scans for hardcoded OpenAI API keys

All tests passing indicates that, for the covered scenarios and current environment, the system is behaving correctly and non-defaulting.

### 7.2 Folder-Specific Tests (Optional)

- Backend only:

```bash
cd RAG-LLM-for-Ports-backend
python test_comprehensive.py
```

- DB setup helpers:

```bash
cd RAG-LLM-for-Ports-backend_dbsetup
python test_comprehensive.py
```

- Translation / multilingual helpers:

```bash
cd RAG-LLM-for-Ports-backend_translate
python test_comprehensive.py
```

---

## 8. Non-Defaulting & Error Behavior

Throughout the codebase:

- RAG and workflow components **raise explicit errors** (e.g. `ValueError`, `HTTPException`) when:
  - No relevant documents are found
  - Recommendations or answers would be empty
  - Decision gates fail or encounter unexpected issues
- The API surfaces these as **clear 4xx/5xx responses** with specific messages.

This ensures that **if the system cannot answer from its data**, it tells you that plainly rather than returning a generic or made‑up response.



