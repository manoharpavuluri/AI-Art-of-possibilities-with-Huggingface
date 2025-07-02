# AI Art of Possibilities with Hugging Face

This repository showcases various AI capabilities using Hugging Face transformers and LangGraph, demonstrating the art of possibilities in artificial intelligence.

## 📚 Contents

### Main Notebook
- **`Art_of_possibilities_with_Huggingface.ipynb`** - Comprehensive notebook covering:
  - Sentiment Analysis using different models
  - Language Translation with T5 models
  - Question-Answering with RoBERTa
  - Data visualization and analysis

### LangGraph Examples
- **`LangGraph/Playing-With-Langgraph1.ipynb`** - LangGraph workflow examples:
  - Basic function chaining workflows
  - LLM-powered topic classification
  - Document processing with vector databases
  - Integration with Groq LLM API

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/manoharpavuluri/AI-Art-of-possibilities-with-Huggingface.git
cd AI-Art-of-possibilities-with-Huggingface
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

### Environment Setup

For optimal performance, consider using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🔧 Features

### Sentiment Analysis
- Default DistilBERT model for general sentiment
- Twitter-specific RoBERTa model for social media analysis
- Batch processing capabilities

### Language Translation
- T5-based translation models
- Support for multiple language pairs
- Specific language direction control

### Question Answering
- RoBERTa-based QA models
- Document-based question answering
- Context-aware responses

### LangGraph Workflows
- Function chaining and workflow orchestration
- LLM integration with Groq API
- Document processing pipelines
- Vector database integration with Chroma

## 📊 Usage Examples

### Sentiment Analysis
```python
from transformers import pipeline

# Default sentiment analysis
sentiment_model = pipeline("sentiment-analysis")
result = sentiment_model("This movie is amazing!")

# Twitter-specific sentiment
twitter_model = pipeline("sentiment-analysis", 
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

### Translation
```python
# English to French translation
translator = pipeline("translation_en_to_fr", model="google-t5/t5-small")
translated = translator("Hello, how are you?")
```

### LangGraph Workflow
```python
from langgraph.graph import Graph

# Create a simple workflow
workflow = Graph()
workflow.add_node("process", your_function)
workflow.set_entry_point("process")
app = workflow.compile()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Groq API](https://console.groq.com/)

## 📞 Contact

For questions or support, please open an issue on GitHub.