# README.md
"""
# AI Port Decision-Support System

An intelligent decision-support system for port operations using LangChain and LangGraph, featuring multilingual document processing, parallel workflow analysis, and advanced security features.

## 🚢 Overview

This system provides AI-powered decision support for port operations, combining:
- **LangChain** for document processing and RAG (Retrieval-Augmented Generation)
- **LangGraph** for workflow orchestration and parallel branching
- **Multilingual support** with automatic translation and language detection
- **Advanced security** with comprehensive data redaction
- **Decision gates** for information validation and quality control

## 🏗️ Architecture

### Core Components

1. **Port Information Retrieval** (`src/port_information_retrieval.py`)
   - ChromaDB vector storage integration
   - Specialized maritime document processing
   - Automatic data redaction for sensitive information
   - Intelligent query processing with source citation

2. **LangGraph Workflow Orchestration** (`src/graph/port_graph.py`)
   - Parallel branching for what-if scenario analysis
   - Decision gates for freshness, confidence, and compliance
   - Workflow state management with checkpointing
   - Asynchronous execution with error handling

3. **Intent Classification** (`src/routers/intent_router.py`)
   - Rule-based question classification
   - Port-specific intent routing
   - Priority-based processing
   - Custom intent pattern support

4. **Decision Nodes** (`src/routers/decision_nodes.py`)
   - Freshness validation for time-sensitive information
   - Confidence scoring for retrieved documents
   - Compliance checking for safety and regulatory requirements
   - Configurable thresholds and rules

5. **FastAPI Server** (`src/api/server.py`)
   - RESTful API endpoints for all operations
   - Comprehensive error handling and logging
   - Health monitoring and system status
   - CORS support for web integration

6. **Multilingual Processing** (`src/utils/loaders.py`)
   - PDF document loading with language detection
   - Automatic translation using MarianMT
   - Port-specific metadata extraction
   - Intelligent document chunking

7. **Advanced Redaction** (`src/utils/redact.py`)
   - Maritime-specific data redaction (IMO, MMSI, vessel names)
   - PII protection (emails, phone numbers, personal names)
   - Financial data redaction (cargo values, account numbers)
   - Custom redaction rule support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YishuoJiang/RAG-LLM-for-Ports.git
   cd RAG-LLM-for-Ports
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp env_example.txt .env
   # Edit .env with your OpenAI API key and other settings
   ```

4. **Run the demo**
   ```bash
   python src/examples/example_usage.py
   ```

### API Server

Start the FastAPI server:

```bash
python src/api/server.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## 📚 API Endpoints

### Core Operations

- **`POST /ask`** - Simple RAG queries
- **`POST /ask_graph`** - LangGraph workflow queries
- **`POST /what-if`** - What-if scenario analysis
- **`POST /search`** - Document similarity search

### System Management

- **`GET /health`** - Health check
- **`GET /system/status`** - Comprehensive system status
- **`GET /summary`** - Document summary

### Configuration

- **`GET /intent/classify`** - Intent classification
- **`GET /intent/statistics`** - Intent pattern statistics
- **`GET /decision/thresholds`** - Decision gate thresholds
- **`POST /decision/thresholds`** - Update thresholds

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-3.5-turbo

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./storage/chroma
CHROMA_COLLECTION_NAME=port_documents

# Decision Thresholds
FRESHNESS_THRESHOLD_HOURS=24
CONFIDENCE_THRESHOLD=0.7
COMPLIANCE_STRICT_MODE=true

# Multilingual Support
SUPPORTED_LANGUAGES=en,es,fr,de,zh,ja,ko
AUTO_TRANSLATE=false
```

### Port Configuration

Configure port-specific settings:

```python
port_config = {
    "port_name": "Los Angeles Port",
    "port_type": "container",
    "port_code": "USLAX",
    "timezone": "America/Los_Angeles",
    "operational_hours": "24/7",
    "has_weather_station": True,
    "has_ais_system": True
}
```

## 🧠 Workflow Examples

### Simple Query

```python
from src.port_information_retrieval import PortInformationRetrieval

retrieval = PortInformationRetrieval(openai_api_key="your-key")
result = await retrieval.query_port_information(
    question="What are the safety protocols for vessel berthing?",
    max_documents=5
)
```

### LangGraph Workflow

```python
from src.graph.port_graph import PortWorkflowGraph

workflow = PortWorkflowGraph(port_retrieval=retrieval)
result = await workflow.execute_workflow(
    query="How should I handle a berth conflict?",
    user_id="port_operator_1",
    port_config={"port_type": "container"}
)
```

### What-If Scenarios

```python
scenarios = [
    "in normal weather conditions",
    "during a storm warning", 
    "with equipment breakdown"
]

result = await workflow.execute_what_if_scenario(
    base_query="How should I handle vessel berthing?",
    scenario_variations=scenarios
)
```

## 🔒 Security Features

### Data Redaction

The system automatically redacts sensitive information:

- **Maritime Data**: IMO numbers, MMSI, vessel names, berth numbers
- **Personal Information**: Names, emails, phone numbers, SSN
- **Financial Data**: Cargo values, account numbers, invoices
- **Port-Specific**: Container numbers, booking references, customs declarations

### Custom Redaction Rules

```python
from src.utils.redact import RedactionRule, AdvancedRedactor

custom_rule = RedactionRule(
    name="custom_pattern",
    pattern=r"Custom\s*:?\s*([A-Z0-9]+)",
    replacement=r"Custom: [REDACTED]",
    description="Redact custom patterns"
)

redactor = AdvancedRedactor(custom_rules=[custom_rule])
redacted_text, counts = redactor.redact_text(text)
```

## 🌍 Multilingual Support

### Language Detection and Translation

```python
from src.utils.loaders import MultilingualDocumentLoader

loader = MultilingualDocumentLoader(
    supported_languages=["en", "es", "fr", "de", "zh"],
    translation_model_name="Helsinki-NLP/opus-mt-en-mul"
)

documents = loader.load_pdf_documents(
    pdf_directory="./documents",
    language_detection=True,
    auto_translate=True
)
```

## 📊 Monitoring and Analytics

### System Status

```python
# Get comprehensive system status
status = await get_system_status()

# Health check
health = await health_check()

# Intent statistics
stats = intent_router.get_intent_statistics()
```

### Decision Gate Monitoring

```python
# Check decision gate results
freshness_result = await decision_nodes.check_freshness(documents, query)
confidence_result = await decision_nodes.check_confidence(documents, query)
compliance_result = await decision_nodes.check_compliance(documents, query)
```

## 🧪 Testing

Run the comprehensive demo:

```bash
python src/examples/example_usage.py
```

This will demonstrate:
- Document ingestion and processing
- Simple RAG queries
- LangGraph workflow execution
- What-if scenario analysis
- Intent classification
- Decision gate functionality
- Data redaction
- System status monitoring

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/api/server.py"]
```

### Environment Setup

1. Set production environment variables
2. Configure external databases (PostgreSQL, Redis)
3. Set up monitoring and logging
4. Configure security settings
5. Set up backup and recovery


