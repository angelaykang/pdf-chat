# pdf-chat

Chat with your PDF documents using retrieval-augmented generation (RAG). Upload PDFs, and the system extracts text, builds vector embeddings, and uses an LLM to answer natural language questions grounded in your documents.

Supports two backends:
- **OpenAI** — uses OpenAI embeddings + GPT for high-quality answers (requires API key)
- **Open-source** — uses HuggingFace sentence-transformers + a local Mistral 7B GGUF model, fully offline with no API costs


### Team: pylovers
| Name | USC ID |
|------|--------|
| Dylan Chen | 6984540266 |
| Angela Kang | 8957777203 |
| Vincent-Daniel Yun | 4463771151 |

---

## How It Works

1. **Extract** — PDF text is pulled out with PyPDF2 and stored in a local SQLite database
2. **Chunk** — Text is split into overlapping 500-character sections using LangChain's `CharacterTextSplitter`
3. **Embed** — Each chunk is embedded into a vector (OpenAI or HuggingFace) and indexed in a FAISS vector store
4. **Retrieve** — Given a user question, the most relevant chunks are retrieved via similarity search
5. **Answer** — The question + retrieved context are passed to an LLM, which generates a grounded answer
6. **Remember** — Conversation history is maintained so follow-up questions work naturally

---

## Prerequisites

- **Python** 3.10+
- **macOS or Linux** (for `llama-cpp-python`; Windows may need extra build setup)
- **~5 GB disk** for the local GGUF model (open-source mode only)
- **OpenAI API key** (OpenAI mode only)

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your PDFs

Drop any PDF files into the `pdfs/` directory. For this lab, use the materials from the [Drive folder](https://drive.google.com/drive/folders/1hdUoDvtQoFkIbJyUr8kwDQghLNTptoCQ) (e.g. the textbook PDF such as “Ads cookbook” and any installation guides):

```
pdfs/
  Ads cookbook .pdf
  ...
```

### 4. (Open-source mode only) Download the local model

```bash
python download_model.py
```

This downloads Mistral 7B Instruct Q4_K_M (~4.4 GB) into `models/`. Only needs to run once.

---

## Usage

### OpenAI mode

```bash
export OPENAI_API_KEY="sk-..."
python app_p1.py openai
```

### Open-source mode (no API key needed)

```bash
python app_p1.py open_source
```

### Show retrieved source chunks

Append `--sources` to see which text chunks the answer was derived from:

```bash
python app_p1.py open_source --sources
```

Type your questions at the prompt. Type `exit` to quit.

**Example session:**

```
Loading PDFs...
  Reading paper.pdf
  Reading manual.pdf
Chunking text...
Got 247 chunks
Building vector store...
Loading LLM...

Chatbot ready (open-source)
Type a question or 'exit' to quit.

You: What is the main topic of this document?
Bot: The document covers ...

You: exit
Goodbye.
```

---

## Project Structure

```
├── app_p1.py            # Main RAG chatbot pipeline
├── download_model.py    # One-time script to fetch the GGUF model
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── pdfs/                # Place your PDF documents here
├── webapp/              # ChatBot Application files
└── models/              # Local LLM model (auto-created by download_model.py)

```
<br><br>

# ChatBot Web Application

First of all, please go to webapp directory and start python flast
```
python web_app.py
```
Then open a browser and go to
```
http://127.0.0.1:5000/
```

<br><br><br><br>
---

## Configuration

These constants at the top of `app_p1.py` can be tuned:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between consecutive chunks to preserve context |
| `K_RETRIEVE` | 12 | Number of top chunks retrieved per query |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model (open-source mode) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |

---

## Environment Variables

| Variable | Required for | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | OpenAI mode | Your OpenAI API key |

Open-source mode does not require any environment variables.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `OPENAI_API_KEY is not set` | Export your key: `export OPENAI_API_KEY="sk-..."` |
| `429` / quota error | Your OpenAI account has no credits — use open-source mode instead |
| Model file not found | Run `python download_model.py` first |
| `llama-cpp-python` won't install | macOS: run `xcode-select --install`. Linux: install `cmake` and `build-essential` |
| No text extracted from PDFs | Make sure your PDFs contain selectable text, not scanned images |
| Download script fails | Check your internet connection and retry. You can also download the model manually from the URL printed by the script and place it in `models/` |

---

## Built With

- [LangChain](https://github.com/langchain-ai/langchain) — text splitting, vector stores, conversation chains
- [FAISS](https://github.com/facebookresearch/faiss) — fast similarity search
- [PyPDF2](https://github.com/py-pdf/pypdf) — PDF text extraction
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — local GGUF model inference
- [sentence-transformers](https://www.sbert.net/) — open-source text embeddings
- [OpenAI API](https://platform.openai.com/) — embeddings and chat completions (optional)
