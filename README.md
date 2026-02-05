# Docling RAG

A completely **free, local RAG (Retrieval-Augmented Generation)** system that answers questions about your PDFs, including scanned documents with automatic OCR.

## 🎯 Features

- 📄 **PDF Processing**: Text extraction + automatic OCR for scanned documents
- 🔍 **Semantic Search**: Find relevant content by meaning, not just keywords
- 🤖 **Local LLM**: Private, offline AI powered by Ollama (no API costs)
- ⚡ **Fast**: FAISS vector database for instant retrieval
- 💰 **100% Free**: No subscriptions, no API calls, no cloud services
- 🔒 **Privacy**: All data stays on your machine

## 🏗️ Architecture

```
PDFs → Docling (OCR) → SentenceTransformers (embeddings) → FAISS (vector DB) → Ollama/Phi-3 → Answers
```

## 📦 Tech Stack

- **Docling**: PDF processing with automatic OCR
- **SentenceTransformers**: Semantic embeddings
- **FAISS**: Vector similarity search
- **Ollama + Phi-3**: Local LLM inference
- **Rich**: Beautiful terminal UI

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - [Download here](https://ollama.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/Supl3x/Docling-RAG.git
cd Docling-RAG

# Install dependencies
pip install -r requirements.txt

# Install Ollama model
ollama pull phi3
```

### Usage

```bash
# 1. Add PDFs to data/pdfs/ folder

# 2. Run the application
python app.py

# 3. First time: Choose "1" to ingest PDFs
# 4. Then: Choose "2" to chat with your documents
```

## 📁 Project Structure

```
DOCLING RAG/
├── data/pdfs/          # Your PDFs go here
├── src/
│   ├── ingest.py       # PDF → Embeddings → Index
│   ├── retrieve.py     # Vector search engine
│   └── chat.py         # LLM interaction
├── index/              # Vector database (auto-created)
├── app.py              # Main application
└── requirements.txt    # Dependencies
```

## 💡 How It Works

### Phase 1: Ingestion
1. Place PDFs in `data/pdfs/`
2. Docling extracts text (runs OCR if needed)
3. Text is chunked into semantic pieces
4. Chunks are converted to embeddings (vectors)
5. FAISS index is built for fast search

### Phase 2: Q&A (RAG)
1. User asks a question
2. Question is converted to embedding
3. FAISS finds most similar document chunks
4. Context + Question sent to local LLM
5. LLM generates answer using only the context
6. Answer displayed with source citations

## 🎓 What You'll Learn

- Document processing & OCR
- Vector embeddings & semantic search
- RAG (Retrieval-Augmented Generation)
- Vector databases (FAISS)
- Local LLM deployment
- System architecture & design

## 🔧 Configuration

Edit these parameters in the code:

**Embeddings** (`src/ingest.py`, `src/retrieve.py`):
- Model: `all-MiniLM-L6-v2` (fast, 384-dim)
- Chunk size: 500 characters

**LLM** (`src/chat.py`):
- Model: `phi3` (2.2GB, efficient)
- Temperature: 0.1 (focused answers)
- Max tokens: 500

**Search** (`src/retrieve.py`):
- Top-K results: 5 chunks

## 📊 Performance

On a typical laptop:
- **OCR**: 2-5 seconds/page
- **Embedding**: 100-500ms/chunk
- **Search**: <50ms
- **LLM response**: 2-10 seconds

## 🆚 Cost Comparison

| Solution | Our System | Commercial (GPT-4 + Pinecone) |
|----------|------------|------------------------------|
| Setup | Free | Free |
| Monthly | **$0** | $90+ |
| Privacy | Local | Cloud |
| Offline | ✅ Yes | ❌ No |

## 🛠️ Troubleshooting

**No PDFs found**: Add `.pdf` files to `data/pdfs/`

**OCR fails**: Ensure scans are clear and readable

**Ollama error**: Check if Ollama service is running

**Out of memory**: Reduce chunk size or use smaller embedding model

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - Free to use, modify, and distribute

## 🌟 Acknowledgments

- [Docling](https://github.com/DS4SD/docling) - Document processing
- [SentenceTransformers](https://www.sbert.net/) - Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Ollama](https://ollama.com) - Local LLM runtime

## 🚀 Future Enhancements

- [ ] Web interface (Streamlit/Gradio)
- [ ] Multi-language support
- [ ] Image/diagram understanding
- [ ] Table extraction
- [ ] Batch processing
- [ ] Cloud storage integration

---

**Made with ❤️ for learning and privacy**
