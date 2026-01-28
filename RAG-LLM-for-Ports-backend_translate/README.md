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

6. **Enhanced Multilingual Processing** (`src/utils/multilingual_processor.py`)
   - OCR support for scanned documents and images
   - Advanced language detection with confidence scoring
   - Robust translation pipeline with MarianMT models
   - Comprehensive metadata preservation
   - Support for 10+ languages and scripts

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

## 🌍 Multilingual Document Processing

The system includes comprehensive multilingual support for processing documents in multiple languages and scripts.

### Supported Languages

- **English** (en) - Latin script
- **Spanish** (es) - Latin script  
- **French** (fr) - Latin script
- **German** (de) - Latin script
- **Italian** (it) - Latin script
- **Portuguese** (pt) - Latin script
- **Chinese** (zh) - Chinese script
- **Japanese** (ja) - Japanese script (Hiragana, Katakana, Kanji)
- **Korean** (ko) - Korean script (Hangul)
- **Arabic** (ar) - Arabic script
- **Russian** (ru) - Cyrillic script

### Features

1. **Text Extraction with OCR Support**
   - Automatic document type detection (PDF, scanned PDF, images, text)
   - OCR processing for scanned documents using EasyOCR and Tesseract
   - Support for multiple image formats (JPG, PNG, TIFF, BMP)

2. **Language Detection with Confidence Scoring**
   - Advanced transformer-based language detection (XLM-RoBERTa)
   - Confidence scoring with reliability assessment
   - Fallback mechanisms for robust detection

3. **Machine Translation Pipeline**
   - MarianMT neural machine translation models
   - Multiple language pair models for better accuracy
   - Automatic translation to canonical English for indexing
   - Error handling and fallback mechanisms

4. **Metadata Preservation**
   - Original content preservation alongside translated text
   - Language detection results and confidence scores
   - Translation history and model information
   - Processing timestamps and document integrity

5. **Multilingual Redaction**
   - Script-specific redaction patterns
   - Unicode support for all writing systems
   - Maritime-specific patterns for port operations
   - Cross-script sensitive data detection

### Installation for Multilingual Support

#### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
python setup_multilingual.py
```

This will automatically:
- Check Python version compatibility
- Install system dependencies (Tesseract OCR)
- Create virtual environment
- Install all Python dependencies with correct version constraints
- Verify the installation

#### Option 2: Manual Setup

1. **Create Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install System Dependencies for OCR**

   **On macOS:**
   ```bash
   brew install tesseract
   ```

   **On Ubuntu/Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

3. **Install Python Dependencies with Version Constraints**
   ```bash
   pip install --upgrade pip
   pip install "numpy>=1.24.3,<2.0.0"
   pip install "torch>=2.1.1,<2.6.0"
   pip install "huggingface-hub>=0.16.0,<0.20.0"
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python verify_install.py
   python test_multilingual_system.py
   ```

#### Option 3: Complete Installation Guide

For detailed installation instructions and troubleshooting, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md).

### Quick Start - Multilingual Processing

1. **Run the Comprehensive Demo**
   ```bash
   python multilingual_demo.py
   ```

2. **Test Individual Components**
   ```python
   from src.utils.multilingual_processor import create_enhanced_processor
   
   # Create processor
   processor = create_enhanced_processor(enable_ocr=True)
   
   # Test language detection
   text = "Este es un manual de operaciones portuarias"
   result = processor.detect_language_with_confidence(text)
   print(f"Language: {result.language}, Confidence: {result.confidence}")
   
   # Test translation
   translated, model = processor.translate_text(text, "es", "en")
   print(f"Translated: {translated}")
   ```

3. **Process Documents**
   ```python
   from src.utils.loaders import create_document_loader
   
   # Create enhanced loader
   loader = create_document_loader(
       multilingual=True,
       enhanced=True,
       enable_ocr=True
   )
   
   # Process documents
   documents = loader.load_pdf_documents(
       "path/to/multilingual/documents",
       auto_translate=True
   )
   
   print(f"Processed {len(documents)} document chunks")
   ```

### Advanced Usage

#### Custom Language Detection
```python
from src.utils.multilingual_processor import create_enhanced_processor

processor = create_enhanced_processor(
    supported_languages=["en", "es", "fr", "de", "zh", "ja", "ko"],
    enable_ocr=True,
    chunk_size=1000,
    chunk_overlap=200
)

# Process single document
result = processor.process_document("document.pdf", auto_translate=True)

if result["success"]:
    metadata = result["processing_metadata"]
    print(f"Original Language: {metadata.original_language}")
    print(f"Translation Model: {metadata.translation_model}")
    print(f"OCR Used: {metadata.ocr_used}")
```

#### Multilingual Redaction
```python
from src.utils.redact import create_redactor

# Create multilingual redactor
redactor = create_redactor(
    maritime=True,
    pii=True,
    financial=True,
    multilingual=True
)

# Test multilingual redaction
text = """
Contact: John Smith (john@example.com) - English
Contacto: Juan Pérez (juan@ejemplo.com) - Spanish
اتصل بأحمد محمد - Arabic
"""

redacted, counts = redactor.redact_text(text)
print(f"Redacted: {redacted}")
print(f"Redaction counts: {counts}")
```

#### Batch Processing
```python
# Process multiple documents
file_paths = ["doc1.pdf", "doc2.jpg", "doc3.txt"]
results = processor.process_documents_batch(file_paths, auto_translate=True)

# Get statistics
stats = processor.get_processing_statistics(results)
print(f"Success Rate: {stats['success_rate']:.1%}")
print(f"Languages Detected: {stats['languages_detected']}")
```

### Integration with Existing System

The multilingual processing seamlessly integrates with the existing port information retrieval system:

```python
from src.port_information_retrieval import PortInformationRetrieval

# Initialize with multilingual support
retrieval = PortInformationRetrieval(openai_api_key="your-api-key")

# Process multilingual documents
pdf_directory = "path/to/multilingual/pdfs"
documents = retrieval.process_pdf_documents(pdf_directory)

# Query in different languages
result_en = await retrieval.query_port_information("What are the berthing procedures?")
result_es = await retrieval.query_port_information("¿Cuáles son los procedimientos de atraque?")
```

### Testing

Run the comprehensive test suite:

```bash
# Run all multilingual tests
python -m pytest tests/test_multilingual_processing.py -v

# Run specific test categories
python -m pytest tests/test_multilingual_processing.py::TestEnhancedMultilingualProcessor -v
python -m pytest tests/test_multilingual_processing.py::TestMultilingualRedaction -v
```

### Troubleshooting

#### Common Issues

1. **OCR Not Working**
   ```bash
   # Check Tesseract installation
   tesseract --version
   
   # Test OCR
   python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
   ```

2. **Translation Models Not Loading**
   ```bash
   # Check internet connection for model downloads
   # Models are downloaded automatically on first use
   ```

3. **Language Detection Issues**
   ```python
   # Test with sufficient text content
   text = "This is a longer text sample for better language detection accuracy."
   result = processor.detect_language_with_confidence(text)
   print(f"Confidence: {result.confidence}")
   ```

### Performance Optimization

- **OCR Processing**: Disable OCR for text-based documents to improve speed
- **Translation Models**: Use smaller models for resource-constrained environments
- **Batch Processing**: Process multiple documents together for better efficiency
- **GPU Acceleration**: Enable GPU support for faster OCR and translation

### Quick Reference

| Command | Description |
|---------|-------------|
| `python setup_multilingual.py` | Automated setup and installation |
| `python verify_install.py` | Verify installation and test components |
| `python test_multilingual_system.py` | Comprehensive system test |
| `python multilingual_demo.py` | Run comprehensive demonstration |
| `python -m pytest tests/test_multilingual_processing.py -v` | Run test suite |
| `python -c "from src.utils.multilingual_processor import create_enhanced_processor; print('✅ Ready!')"` | Quick component test |

### File Structure

```
src/utils/
├── multilingual_processor.py    # Core multilingual processing engine
├── redact.py                   # Enhanced multilingual redaction
└── loaders.py                  # Enhanced document loaders

tests/
└── test_multilingual_processing.py  # Comprehensive test suite

multilingual_demo.py            # Complete demonstration script
setup_multilingual.py           # Automated setup script
verify_install.py               # Installation verification
test_multilingual_system.py     # Comprehensive system test
INSTALLATION_GUIDE.md           # Complete installation guide
README_multilingual.md          # Detailed documentation
```

For more detailed information, see [README_multilingual.md](README_multilingual.md).

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


