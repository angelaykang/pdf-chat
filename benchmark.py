import os
import sqlite3
import sys
import time
import json
import glob
import textwrap
import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp


# Base Config Variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50 # overlap keeps context intact across chunk boundaries
K_RETRIEVE = 5 # number of top chunks passed to the LLM as context


# Embedding Models
EMBEDDING_MODELS = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}


# Local LLMs
# (filename, huggingface_repo, hf_filename, n_ctx)
LLM_MODELS = {
    "Mistral-7B": (
        "mistral-7b-instruct.Q4_K_M.gguf",
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        8192,
    ),
    "Phi-3-Mini-3.8B": (
        "Phi-3-mini-4k-instruct-q4.gguf",
        "microsoft/Phi-3-mini-4k-instruct-gguf",
        "Phi-3-mini-4k-instruct-q4.gguf",
        4096,
    ),
    "TinyLlama-1.1B": (
        "tinyllama-1.1b-chat.Q4_K_M.gguf",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        2048,
    ),
}


# Default Text Questions
# Based on analyzing the gdrive PDFs
TEST_QUESTIONS = [
    {
        "question": "How do I create a new workspace in ADS?",
        "expected_keywords": ["workspace", "launch", "create", "ADS"],
    },
    {
        "question": "What is Harmonic Balance simulation used for?",
        "expected_keywords": ["harmonic", "balance", "nonlinear", "frequency"],
    },
    {
        "question": "How do you perform tuning in ADS?",
        "expected_keywords": ["tuning", "tune", "component", "slider"],
    },
    {
        "question": "What is S-parameter simulation?",
        "expected_keywords": ["s-parameter", "port", "linear", "frequency"],
    },
    {
        "question": "How do you set up a layout for co-simulation?",
        "expected_keywords": ["layout", "co-simulation", "pin", "schematic"],
    },
    {
        "question": "What are CPW transmission lines?",
        "expected_keywords": ["cpw", "coplanar", "waveguide", "ground"],
    },
    {
        "question": "How does the optimization feature work in ADS?",
        "expected_keywords": ["optim", "goal", "variable", "gradient"],
    },
    {
        "question": "What is FlexNet licensing in ADS?",
        "expected_keywords": ["flexnet", "license", "server", "admin"],
    },
    {
        "question": "How do you install an ADS license?",
        "expected_keywords": ["license", "install", "file", "server"],
    },
    {
        "question": "What types of simulation does ADS support?",
        "expected_keywords": ["simulation", "harmonic", "transient", "linear"],
    },
]


# Reusing some PDF extraction logic/functions from app_p1.py for consistency
# Function to read all PDF texts
def extract_all_pdfs(folder):
    all_text = []
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        reader = PdfReader(path)
        for page in reader.pages:
            try:
                text = page.extract_text()
            except Exception:
                text = ""
            all_text.append(text or "")
    return "\n".join(all_text)

# Text chunking function
def chunk_text(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)


# Embedding Model Comparisons
# Fraction of expected keywords found in retrieved chunks
def keyword_hit_rate(retrieved_chunks, keywords):
    combined = " ".join(retrieved_chunks).lower()
    hits = sum(1 for kw in keywords if kw.lower() in combined)

    return hits / len(keywords) if keywords else 0.0

# Building vector store per embedding model and evaluating retrieval quality
def benchmark_embeddings(chunks):
    rows = []
 
    # iterating through embedding models
    for model_label, model_name in EMBEDDING_MODELS.items():
        print(f"Embedding Model: {model_label}")
 
        # building vector store (and logging time to completion)
        t0 = time.time()
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        build_time = time.time() - t0
        print(f"Vector Store built in {build_time:.2f}s ({len(chunks)} chunks)")

        # evaluating retrieval per question
        for tq in TEST_QUESTIONS:
            q = tq["question"]
            # get results + time logging
            t1 = time.time()
            results = vector_store.similarity_search_with_score(q, k=K_RETRIEVE)
            query_time = time.time() - t1

            retrieved_texts = [doc.page_content for doc, _score in results]
            l2_distances = [float(score) for _doc, score in results]

            # cosine similarity for normalised embeddings
            cosine_sims = [1.0 - (d ** 2) / 2.0 for d in l2_distances]
            avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0

            # calculating chunked percent of keyword presence
            hit_rate = keyword_hit_rate(retrieved_texts, tq["expected_keywords"])

            rows.append({
                "embedding_model": model_label,
                "question": q,
                "avg_cosine_sim": round(avg_cosine, 4),
                "keyword_hit_rate": round(hit_rate, 4),
                "query_time_s": round(query_time, 4),
                "build_time_s": round(build_time, 2),
            })

            # printing results in format
            short_q = q[:50]
            print(f" - Q: {short_q:<52} cos={avg_cosine:.3f}  hits={hit_rate:.0%}")

    df = pd.DataFrame(rows)
    return df

# Create aggregated table of per-model results
def summarise_embeddings(df):
    if df.empty:
        return df
    summary = (
        df.groupby("embedding_model").agg(
            avg_cosine_sim=("avg_cosine_sim", "mean"),
            avg_keyword_hit_rate=("keyword_hit_rate", "mean"),
            avg_query_time_s=("query_time_s", "mean"),
            build_time_s=("build_time_s", "first"),
        ).round(4).sort_values("avg_keyword_hit_rate", ascending=False)
    )
    return summary


# Local LLM Comparisons
# Setting a default prompt to use for comparisons
PROMPT_TEMPLATE = textwrap.dedent("""\
    Use the following context extracted from PDF documents to answer the
    question.  If the context does not contain enough information, say so.
    Keep your answer concise (3-5 sentences).
 
    Context:
    {context}
 
    Question: {question}
 
    Answer:"""
)

# Function to build FAISS store with the baseline MiniLM embedding model
def build_default_vector_store(chunks):
    embed_model = list(EMBEDDING_MODELS.values())[0]
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.from_texts(chunks, embedding=embeddings)

# LLM Benchmarking function
# Runs each LLM on the test questions and measures quality and speed
def benchmark_llms(chunks, embed_model_key):
    # Use specified embedding model or default to MiniLM
    if embed_model_key and embed_model_key in EMBEDDING_MODELS:
        emb_name = EMBEDDING_MODELS[embed_model_key]
    else:
        emb_name = list(EMBEDDING_MODELS.values())[0]
        embed_model_key = list(EMBEDDING_MODELS.keys())[0]

    # Make vector store
    print(f"\nBuilding shared vector store with {embed_model_key}")
    embeddings = HuggingFaceEmbeddings(model_name=emb_name)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    rows = []

    # Iterating through LLMs
    for llm_label, (filename, _repo, _hf_file, n_ctx) in LLM_MODELS.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if not os.path.isfile(model_path):
            print(f"Skipping {llm_label}: model file not found at {model_path}")
            print(f"Run: python benchmark.py --download-models")
            continue

        print(f"LLM: {llm_label} ({filename})")

        # Load LLM and log time
        t0 = time.time()
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=512,
            n_ctx=n_ctx,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        load_time = time.time() - t0
        model_size_mb = os.path.getsize(model_path) / (1024 ** 2)
        print(f"Loaded in {load_time:.2f}s ({model_size_mb:.0f} MB)")

        # Iterating through default text questions
        for tq in TEST_QUESTIONS:
            q = tq["question"]

            # retrieving context
            docs = vector_store.similarity_search(q, k=K_RETRIEVE)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = PROMPT_TEMPLATE.format(context=context, question=q)

            # generating answer and logging time
            t1 = time.time()
            try:
                answer = llm.invoke(prompt)
                answer = answer.strip()
                error = ""
            except Exception as e:
                answer = ""
                error = str(e)
            # log latency
            latency = time.time() - t1

            # store results
            rows.append({
                "llm_model": llm_label,
                "question": q,
                "answer": answer[:500],
                "latency_s": round(latency, 2),
                "load_time_s": round(load_time, 2),
                "model_size_mb": round(model_size_mb, 1),
                "error": error,
            })

            # printing q&a
            short_q = q[:50]
            short_a = (answer[:75] + "...") if len(answer) > 75 else answer
            print(f"    Q: {short_q:<52} ({latency:.2f}s)")
            print(f"    A: {short_a}")

        # free memory before loading next model
        del llm

    return pd.DataFrame(rows)

# Create aggregated table of per-model results
def summarise_llms(df):
    if df.empty:
        return df
    summary = (
        df.groupby("llm_model").agg(
            avg_latency_s=("latency_s", "mean"),
            model_size_mb=("model_size_mb", "first"),
            load_time_s=("load_time_s", "first"),
            questions_answered=("answer", lambda s: sum(1 for a in s if a)),
        ).round(2).sort_values("avg_latency_s")
    )
    return summary


# Downloading additional models function
# Phi-3 Mini, TinyLlama GGUFs
def download_extra_models():
    # going to use huggingface_hub for model downloads
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    # iterating through model list
    for label, (filename, repo, hf_file, _ctx) in LLM_MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.isfile(dest):
            print(f"{label} already present ({dest})")
            continue

        # model downloading
        print(f"\nDownloading {label} from {repo}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=hf_file,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may nest file
            # move if needed
            if os.path.abspath(downloaded) != os.path.abspath(dest):
                os.rename(downloaded, dest)
            size_gb = os.path.getsize(dest) / (1024 ** 3)
            print(f"Saved {label} to {dest} ({size_gb:.1f} GB)")
        except Exception as e:
            print(f"Failed to download {label}: {e}")


# Main function
def main():
    # get CLI arguments
    args = [a.lower() for a in sys.argv[1:]]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if "--download-models" in args:
        download_extra_models()
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # extracting text
    print(f"Extracting text from PDFs in {PDF_FOLDER}")
    raw_text = extract_all_pdfs(PDF_FOLDER)
    if not raw_text.strip():
        print("No text extracted. Add PDFs to pdfs/ and retry.")
        return

    # chunking text
    chunks = chunk_text(raw_text)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    # get run configuration
    run_embed = args[0] in ("embeddings", "all")
    run_llms = args[0] in ("llms", "all")

    # embedding comparison run
    if run_embed:
        print("Embedding Model Comparison:")
        embed_df = benchmark_embeddings(chunks)

        # save results to specified path
        embed_path = os.path.join(RESULTS_DIR, "embedding_results.csv")
        embed_df.to_csv(embed_path, index=False)
        print(f"\nEmbedding results saved to {embed_path}")

        # print and save summary
        summary = summarise_embeddings(embed_df)
        print("\nEmbedding Summary")
        print(summary.to_string())
        summary_path = os.path.join(RESULTS_DIR, "embedding_summary.csv")
        summary.to_csv(summary_path)

    # LLM comparison run
    if run_llms:
        print("Local LLM Comparisons:")
        llm_df = benchmark_llms(chunks)

        # validate model existence
        if llm_df.empty:
            print("\nNo LLM models found. Run: python benchmark.py --download-models")
        else:
            # save results to specified path
            llm_path = os.path.join(RESULTS_DIR, "llm_results.csv")
            llm_df.to_csv(llm_path, index=False)
            print(f"\nLLM results saved to {llm_path}")

            # print and save summary
            summary = summarise_llms(llm_df)
            print("\nLLM Summary")
            print(summary.to_string())
            summary_path = os.path.join(RESULTS_DIR, "llm_summary.csv")
            summary.to_csv(summary_path)

if __name__ == "__main__":
    main()
