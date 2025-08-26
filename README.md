# 📄 AI Resume Parser & Matcher

An intelligent resume parsing and matching application built with Streamlit that uses local LLMs via Ollama to analyze resumes against job descriptions and provide detailed hiring recommendations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Multi-format Support**: Parse PDF, Word documents, and text files
- **Flexible Input Methods**: Load job descriptions from files, upload new ones, or paste directly
- **AI-Powered Analysis**: Uses local LLMs for intelligent resume-job matching
- **Structured Output**: Provides match scores, strengths, gaps, and specific recommendations
- **Analysis History**: Track previous analyses in an intuitive sidebar
- **Download Results**: Export analysis reports for documentation
- **Configurable Models**: Choose from multiple embedding and language models
- **Real-time Processing**: Instant text extraction and embedding generation

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running locally
3. Required Ollama models downloaded (see [Model Setup](#model-setup))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-resume-parser.git
cd ai-resume-parser
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📦 Dependencies

Create a `requirements.txt` file with the following packages:

```
streamlit>=1.28.0
unstructured[all-docs]>=0.10.0
langchain-ollama>=0.1.0
langchain-community>=0.0.20
langchain-core>=0.1.0
faiss-cpu>=1.7.4
python-magic-bin>=0.4.14  # Windows only
```

## 🔧 Model Setup

### Install Ollama

**Windows/Mac/Linux:**
```bash
# Visit https://ollama.ai to download the installer
# Or use curl (Linux/Mac):
curl -fsSL https://ollama.ai/install.sh | sh
```

### Download Required Models

```bash
# Essential models
ollama pull nomic-embed-text    # For embeddings
ollama pull llama3              # For analysis (default)

# Optional alternative models
ollama pull all-minilm          # Alternative embedding model
ollama pull mxbai-embed-large   # High-quality embeddings
ollama pull llama2              # Alternative LLM
ollama pull mistral             # Lightweight LLM
ollama pull codellama           # Code-focused LLM
```

## 📖 Usage Guide

### 1. Job Description Setup

Choose one of three methods to input job descriptions:

**Method A: File-based (Recommended)**
- Place your job description in `file.txt` in the project root
- The app will automatically load it on startup

**Method B: Upload**
- Use the file uploader to select PDF, Word, or text files
- Supports drag-and-drop functionality

**Method C: Direct Input**
- Paste job description text directly into the text area
- Ideal for quick testing or when copying from web sources

### 2. Resume Upload

- Click "Upload resume" and select your file
- Supported formats: PDF, DOCX, DOC, TXT
- The app will extract and preview the text content

### 3. AI Analysis

- Click "🚀 Analyze Resume Match" to start the analysis
- Wait for the AI to process both documents
- Review the structured analysis results

### 4. Results Management

- **View Results**: Detailed analysis with scores and recommendations
- **Download**: Save analysis as a text file for records
- **History**: Access previous analyses in the sidebar
- **Reset**: Clear all data to start fresh

## 🎯 Analysis Output

The AI provides a comprehensive analysis including:

- **🎯 RECOMMENDATION**: SHORTLISTED or NOT SHORTLISTED
- **📊 MATCH SCORE**: Numerical rating from 1-10
- **💪 KEY STRENGTHS**: Relevant skills and experiences
- **⚠️ GAPS IDENTIFIED**: Missing requirements or weak areas
- **📋 SPECIFIC EXAMPLES**: Direct citations from the resume
- **📝 SUMMARY**: Overall assessment and reasoning

## ⚙️ Configuration

### Model Selection

Use the sidebar to configure:
- **Embedding Models**: Choose text processing model
- **Language Models**: Select the LLM for analysis
- **Analysis Parameters**: Adjust chunk sizes and overlap

### Performance Tuning

**For better performance:**
- Use `mxbai-embed-large` for higher quality embeddings
- Use `llama2` or `mistral` for faster processing
- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in the code

**For resource-constrained systems:**
- Use `all-minilm` for lightweight embeddings
- Reduce chunk sizes to save memory
- Close other applications to free up RAM

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Text Extraction │    │   Vector Store  │
│                 │───▶│                  │───▶│                 │
│ File Upload     │    │ Unstructured     │    │ FAISS + Ollama  │
│ Text Input      │    │ PDF/Word/TXT     │    │ Embeddings      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Analysis      │    │     Ollama       │             │
│   Results       │◀───│      LLM         │◀────────────┘
│                 │    │                  │
│ Match Score     │    │ Local Processing │
│ Recommendations │    │ No Data Sent     │
└─────────────────┘    └──────────────────┘
```

## 🔍 Troubleshooting

### Common Issues

**❌ "Ollama connection failed"**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve
```

**❌ "Model not found"**
```bash
# Download missing models
ollama pull nomic-embed-text
ollama pull llama3
```

**❌ "Text extraction failed"**
- Ensure the uploaded file is not corrupted
- Try converting PDF to text format
- Check file permissions

**❌ "Memory issues with large files"**
- Reduce `CHUNK_SIZE` in the configuration
- Close other applications
- Consider using a more powerful machine

### Performance Issues

**Slow processing:**
- Switch to lighter models (`mistral` instead of `llama3`)
- Reduce document size by preprocessing
- Increase system RAM or use cloud deployment

**High memory usage:**
- Clear analysis history regularly
- Reset the app between large batch processing
- Monitor system resources

## 🧪 Advanced Usage

### Batch Processing

For processing multiple resumes:

1. Create a script wrapper around the core functions
2. Use the `analyze_resume()` function directly
3. Process files in batches to manage memory

### Custom Models

To use custom Ollama models:

1. Add your model to the dropdown options in `main()`
2. Ensure the model is pulled: `ollama pull your-model-name`
3. Test compatibility with the existing prompt structure

### API Integration

The core functions can be extracted for API usage:

```python
from your_app import extract_text, create_vectorstore, analyze_resume

# Use in your own applications
text = extract_text(file)
vectorstore = create_vectorstore(text)
result = analyze_resume(job_desc, resume_text)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Commit** your changes: `git commit -am 'Add feature'`
4. **Push** to the branch: `git push origin feature-name`
5. **Submit** a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/mohit26061999/ai-resume-parser.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 app.py
```

## 🙏 Acknowledgments

- **[Ollama](https://ollama.ai/)** - Local LLM runtime
- **[LangChain](https://langchain.com/)** - LLM application framework
- **[Streamlit](https://streamlit.io/)** - Web app framework
- **[Unstructured](https://unstructured.io/)** - Document processing
- **[FAISS](https://faiss.ai/)** - Vector similarity search

## 📞 Support
- **Email**: mk079823@gmail.com

## 🔮 Roadmap

- [ ] **Multi-language support** for international resumes
- [ ] **Batch processing interface** for HR teams
- [ ] **Custom scoring criteria** configuration
- [ ] **Resume improvement suggestions**
- [ ] **Integration with ATS systems**
- [ ] **Resume template matching**
- [ ] **Skills gap analysis visualization**
- [ ] **Interview question generation**

---

⭐ **If this project helped you, please give it a star on GitHub!**

Made with ❤️ by Mohit Kumar (https://github.com/mohit26061999)


