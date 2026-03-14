import os
import sqlite3
import sys
import glob

from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "pdf_chatbot.db")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50      # overlap keeps context intact across chunk boundaries
K_RETRIEVE = 12         # number of top chunks passed to the LLM as context

LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "qwen2.5-3b-instruct.Q4_K_M.gguf")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"


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
        pages.append(text or "")  # guard against None from encrypted/image-only pages
    return "\n".join(pages)


def load_all_pdfs(folder):
    init_db()
    all_texts = []
    conn = sqlite3.connect(DB_PATH)
    try:
        for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
            print(f"  Reading {os.path.basename(path)}")
            try:
                text = extract_text_from_pdf(path)
                if text.strip():
                    all_texts.append(text)
                    # INSERT OR REPLACE so re-running updates stale entries
                    conn.execute(
                        "INSERT OR REPLACE INTO pdf_extractions (pdf_path, full_text) VALUES (?, ?)",
                        (path, text),
                    )
            except Exception as e:
                print(f"  Could not read {path}: {e}")
        conn.commit()
    finally:
        conn.close()
    return "\n".join(all_texts)


def get_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP, length_function=len
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
            max_tokens=4096,
        )
    else:
        if not os.path.isfile(LOCAL_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {LOCAL_MODEL_PATH}. Run python download_model.py first."
            )
        return LlamaCpp(
            model_path=LOCAL_MODEL_PATH,
            temperature=0.1,
            max_tokens=4096,
            n_ctx=16384,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )


def create_conversation_chain(llm, vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": K_RETRIEVE}),
        memory=memory,
        return_source_documents=True,  # exposes retrieved chunks in the result dict
    )


def run_chatbot(mode, show_sources=False):
    os.makedirs(PDF_FOLDER, exist_ok=True)
    if not glob.glob(os.path.join(PDF_FOLDER, "*.pdf")):
        print(f"No PDFs found in {PDF_FOLDER}. Add some PDF files and try again.")
        return

    print("Loading PDFs...")
    raw_text = load_all_pdfs(PDF_FOLDER)
    if not raw_text.strip():
        print("No text found in PDFs.")
        return

    print("Chunking text...")
    chunks = get_chunks(raw_text)
    print(f"Got {len(chunks)} chunks")

    try:
        print("Building vector store...")
        vector_store = build_vector_store(chunks, mode)
        print("Loading LLM...")
        llm = get_llm(mode)
    except Exception as e:
        print("Error setting up:", e)
        return

    chain = create_conversation_chain(llm, vector_store)

    label = "OpenAI" if mode == "openai" else "open-source"
    print(f"\nChatbot ready ({label})")
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
            result = chain({"question": question})
            answer = result["answer"]
            docs = result.get("source_documents", [])
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print("Bot:", answer)
        if show_sources:
            for i, doc in enumerate(docs, 1):
                snippet = doc.page_content[:100].replace("\n", " ")
                print(f"  [{i}] {snippet}...")
        print()


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python app_p1.py openai|open_source [--sources]")
        return

    mode = args[0].lower()
    if mode not in ("openai", "open_source"):
        print("Mode must be 'openai' or 'open_source'")
        return

    show_sources = "--sources" in args or "-s" in args
    run_chatbot(mode, show_sources)


if __name__ == "__main__":
    main()
