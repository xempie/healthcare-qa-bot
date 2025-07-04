# Healthcare Q&A Bot

A comprehensive healthcare question-answering system built with LangChain, featuring AI agents, document processing, and safety-first design for medical information assistance.

## Features

- **Medical Document Processing**: Automatic loading and processing of PDFs, DOCX, TXT medical documents
- **Intelligent Q&A**: Advanced question-answering using LangChain chains and OpenAI models
- **Medical Agents**: Specialized LangChain agents with healthcare-specific tools
- **Safety Features**: Emergency detection, content filtering, and medical disclaimers
- **Vector Search**: Efficient document retrieval using ChromaDB or FAISS
- **Web Interface**: Interactive Streamlit web application
- **CLI Interface**: Command-line interface for direct interaction

## Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/xempie/healthcare-qa-bot.git
cd healthcare-qa-bot
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. Add medical documents to `data/documents/` directory

4. Initialize the system:
```bash
cd src
python main.py --rebuild
```

### Usage

**Web Interface:**
```bash
streamlit run streamlit_app/app.py
```

**CLI Interface:**
```bash
# Interactive mode
python src/main.py --interactive

# Single question
python src/main.py --question "What are the symptoms of diabetes?"
```

## Configuration

Key environment variables in `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
VECTOR_STORE_TYPE=chromadb
EMBEDDING_CHUNK_SIZE=1000
ENABLE_MEDICAL_DISCLAIMER=True
```

## Project Structure

```
healthcare-qa-bot/
├── src/
│   ├── main.py                 # Main application
│   ├── agents/                 # Medical AI agents
│   ├── chains/                 # Q&A chains
│   ├── document_processing/    # Document loading and processing
│   ├── prompts/               # Medical prompt templates
│   └── utils/                 # Validation and utilities
├── streamlit_app/             # Web interface
├── data/
│   ├── documents/             # Medical documents
│   └── vectorstore/          # Vector store data
├── config/                    # Configuration
├── tests/                     # Test suite
└── docker/                    # Docker deployment
```

## Medical Agents and Tools

The system includes specialized tools:

- **Safety Check Tool**: Emergency detection and safety classification
- **Medical Search Tool**: General medical information retrieval
- **FAQ Search Tool**: Frequently asked questions
- **Specialty Search Tool**: Search by medical specialty
- **Terminology Tool**: Medical term explanations

## Safety Features

- **Emergency Detection**: Automatic detection of medical emergencies with immediate redirection
- **Content Filtering**: Identification of queries requiring professional medical consultation
- **Input Validation**: Comprehensive validation of user queries
- **Medical Disclaimers**: Appropriate disclaimers based on query content

## Supported Document Formats

- PDF: Medical guidelines, research papers
- DOCX: Clinical documents, policies
- TXT: Plain text medical information
- PPTX: Medical presentations
- XLSX: Medical data tables

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up
```

## Testing

```bash
# Run test suite
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Usage Examples

```python
from src.main import initialize_bot

# Initialize the bot
bot = await initialize_bot()

# Ask a question
result = bot.ask_question("What are the symptoms of hypertension?")
print(result['response'])

# Use medical agent
result = bot.ask_question(
    "Explain diabetes symptoms and prevention",
    use_agent=True,
    include_sources=True
)

# Add new documents
result = bot.add_documents(['/path/to/medical_document.pdf'])
```

## Important Medical Disclaimer

This system provides general health information for educational purposes only and should not replace professional medical advice. Always consult qualified healthcare professionals for medical concerns, diagnosis, or treatment decisions.

### Emergency Notice
If experiencing a medical emergency:
- Call 911 (US) or local emergency services
- Go to the nearest emergency room
- Contact your healthcare provider immediately

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## Support

- GitHub Issues for bug reports and feature requests
- Check documentation in `/docs` directory
- Review test cases for usage examples
