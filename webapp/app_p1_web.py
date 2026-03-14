import os
import sqlite3
import glob

from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "pdf_chatbot.db")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
K_RETRIEVE = 12

LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "mistral-7b-instruct.Q4_K_M.gguf")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"

USE_GPU = os.getenv("USE_GPU", "1") == "1"
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))
N_BATCH = int(os.getenv("N_BATCH", "512"))
N_CTX = int(os.getenv("N_CTX", "16384"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
F16_KV = os.getenv("F16_KV", "1") == "1"
VERBOSE = os.getenv("LLAMA_VERBOSE", "1") == "1"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pdf_extractions (
            pdf_path TEXT PRIMARY KEY,
            full_text TEXT NOT NULL,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.commit()
    conn.close()


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = ""
        pages.append(text or "")
    return "\n".join(pages)


def load_all_pdfs(folder):
    init_db()
    all_texts = []
    conn = sqlite3.connect(DB_PATH)

    try:
        for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
            print(f"Reading {os.path.basename(path)}")
            try:
                text = extract_text_from_pdf(path)
                if text.strip():
                    all_texts.append(text)
                    conn.execute(
                        "INSERT OR REPLACE INTO pdf_extractions (pdf_path, full_text) VALUES (?, ?)",
                        (path, text),
                    )
            except Exception as e:
                print(f"Could not read {path}: {e}")
        conn.commit()
    finally:
        conn.close()

    return "\n".join(all_texts)


def get_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)


def build_vector_store(chunks, mode):
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    return FAISS.from_texts(chunks, embedding=embeddings)


def get_llm(mode):
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            openai_api_key=api_key,
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=1024,
        )

    if not os.path.isfile(LOCAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {LOCAL_MODEL_PATH}. Put your GGUF model there first."
        )

    llama_kwargs = dict(
        model_path=LOCAL_MODEL_PATH,
        temperature=0.1,
        max_tokens=1024,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        f16_kv=F16_KV,
        verbose=VERBOSE,
    )

    if USE_GPU:
        llama_kwargs["n_gpu_layers"] = N_GPU_LAYERS
        print(f"Using local model with GPU enabled (n_gpu_layers={N_GPU_LAYERS})")
    else:
        print("Using local model in CPU mode")

    return LlamaCpp(**llama_kwargs)


def create_conversation_chain(llm, vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": K_RETRIEVE}),
        memory=memory,
        return_source_documents=True,
    )


def build_chatbot(mode="open_source"):
    os.makedirs(PDF_FOLDER, exist_ok=True)

    pdf_paths = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {PDF_FOLDER}")

    raw_text = load_all_pdfs(PDF_FOLDER)
    if not raw_text.strip():
        raise ValueError("No text found in PDFs.")

    chunks = get_chunks(raw_text)
    vector_store = build_vector_store(chunks, mode)
    llm = get_llm(mode)
    chain = create_conversation_chain(llm, vector_store)

    return chain, len(chunks), len(pdf_paths)


def ask_question(chain, question, show_sources=False):
    result = chain({"question": question})
    answer = result["answer"]
    docs = result.get("source_documents", [])

    sources = []
    if show_sources:
        for i, doc in enumerate(docs, 1):
            snippet = doc.page_content[:200].replace("\n", " ")
            sources.append(
                {
                    "index": i,
                    "snippet": snippet,
                }
            )

    return {
        "answer": answer,
        "sources": sources,
    }


def run_chatbot_cli(mode="open_source", show_sources=False):
    try:
        chain, chunk_count, pdf_count = build_chatbot(mode)
    except Exception as e:
        print("Error setting up chatbot:", e)
        return

    label = "OpenAI" if mode == "openai" else "open-source"
    print(f"\nChatbot ready ({label})")
    print(f"Loaded {pdf_count} PDFs and created {chunk_count} chunks.")
    print("Type a question or 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye.")
            break

        try:
            output = ask_question(chain, question, show_sources=show_sources)
            print("Bot:", output["answer"])
            if show_sources:
                for src in output["sources"]:
                    print(f"  [{src['index']}] {src['snippet']}...")
            print()
        except Exception as e:
            print("Error:", e)
            print()


if __name__ == "__main__":
    run_chatbot_cli("open_source", show_sources=True)