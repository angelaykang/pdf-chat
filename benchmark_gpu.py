import os
import sys
import time
import glob
import textwrap
import gc
import warnings
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

os.environ["HF_HOME"] = WEIGHTS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(WEIGHTS_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(WEIGHTS_DIR, "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(
    WEIGHTS_DIR, "sentence_transformers"
)

import pandas as pd
import torch
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=UserWarning)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
K_RETRIEVE = 5

USE_CUDA = os.getenv("USE_CUDA", "1") == "1"
USE_4BIT = os.getenv("USE_4BIT", "1") == "1"
DTYPE_STR = os.getenv("DTYPE", "float16").lower()
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda" if USE_CUDA else "cpu")

N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
DO_SAMPLE = os.getenv("DO_SAMPLE", "0") == "1"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "50"))

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "1") == "1"
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "1800"))

EMBEDDING_MODELS = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}

LLM_MODELS = {
    "Mistral-7B": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "n_ctx": 8192,
    },
    "Qwen2.5-3B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "n_ctx": 32768,
    },
    "TinyLlama-1.1B": {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "n_ctx": 2048,
    },
}

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

PROMPT_TEMPLATE = textwrap.dedent("""\
Use the following context extracted from PDF documents to answer the
question. If the context does not contain enough information, say so.
Keep your answer concise (3-5 sentences).

Context:
{context}

Question: {question}

Answer:
""")


class SentenceTransformerEmbeddings:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        cache_folder: str = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_folder = cache_folder

        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        emb = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return emb.tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


def get_torch_dtype():
    return torch.bfloat16 if DTYPE_STR == "bfloat16" else torch.float16


def print_env_summary():
    print("Runtime Configuration")
    print(f"  USE_CUDA        = {USE_CUDA}")
    print(f"  USE_4BIT        = {USE_4BIT}")
    print(f"  DTYPE           = {DTYPE_STR}")
    print(f"  DEVICE_MAP      = {DEVICE_MAP}")
    print(f"  EMBED_DEVICE    = {EMBED_DEVICE}")
    print(f"  MAX_TOKENS      = {MAX_TOKENS}")
    print(f"  MAX_INPUT_TOKENS= {MAX_INPUT_TOKENS}")
    print(f"  DO_SAMPLE       = {DO_SAMPLE}")
    print(f"  TEMPERATURE     = {TEMPERATURE}")
    print(f"  TOP_P           = {TOP_P}")
    print(f"  TOP_K           = {TOP_K}")
    print(f"  N_THREADS       = {N_THREADS}")
    print(f"  WEIGHTS_DIR     = {WEIGHTS_DIR}")
    print()

    print("Torch / CUDA")
    print(f"  torch.__version__         = {torch.__version__}")
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda device count         = {torch.cuda.device_count()}")
        print(f"  current device            = {torch.cuda.current_device()}")
        print(
            f"  device name               = "
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    print()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_model_size_mb(model):
    try:
        total_bytes = 0
        seen = set()
        for p in model.parameters():
            ptr = p.data_ptr()
            if ptr not in seen:
                total_bytes += p.numel() * p.element_size()
                seen.add(ptr)
        return round(total_bytes / (1024 ** 2), 1)
    except Exception:
        return 0.0


def extract_all_pdfs(folder):
    all_text = []
    pdf_paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))

    if not pdf_paths:
        return ""

    for path in pdf_paths:
        try:
            reader = PdfReader(path)
        except Exception as e:
            print(f"[WARN] Could not open {path}: {e}")
            continue

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
            except Exception as e:
                print(f"[WARN] Failed to extract page {i} in {path}: {e}")
                text = ""
            all_text.append(text or "")

    return "\n".join(all_text)


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)


def keyword_hit_rate(retrieved_chunks, keywords):
    combined = " ".join(retrieved_chunks).lower()
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return hits / len(keywords) if keywords else 0.0


def build_embedding_model(model_name):
    return SentenceTransformerEmbeddings(
        model_name=model_name,
        device=EMBED_DEVICE,
        batch_size=EMBED_BATCH_SIZE,
        cache_folder=os.path.join(WEIGHTS_DIR, "sentence_transformers"),
    )


def benchmark_embeddings(chunks):
    rows = []

    for model_label, model_name in EMBEDDING_MODELS.items():
        print(f"Embedding Model: {model_label}")

        t0 = time.time()
        embeddings = build_embedding_model(model_name)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        build_time = time.time() - t0
        print(f"Vector Store built in {build_time:.2f}s ({len(chunks)} chunks)")

        for tq in TEST_QUESTIONS:
            q = tq["question"]

            t1 = time.time()
            results = vector_store.similarity_search_with_score(q, k=K_RETRIEVE)
            query_time = time.time() - t1

            retrieved_texts = [doc.page_content for doc, _score in results]
            l2_distances = [float(score) for _doc, score in results]

            cosine_sims = [1.0 - (d ** 2) / 2.0 for d in l2_distances]
            avg_cosine = (
                sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
            )
            hit_rate = keyword_hit_rate(retrieved_texts, tq["expected_keywords"])

            rows.append(
                {
                    "embedding_model": model_label,
                    "question": q,
                    "avg_cosine_sim": round(avg_cosine, 4),
                    "keyword_hit_rate": round(hit_rate, 4),
                    "query_time_s": round(query_time, 4),
                    "build_time_s": round(build_time, 2),
                }
            )

            short_q = q[:50]
            print(f" - Q: {short_q:<52} cos={avg_cosine:.3f}  hits={hit_rate:.0%}")

        del vector_store
        del embeddings
        clear_memory()

    return pd.DataFrame(rows)


def summarise_embeddings(df):
    if df.empty:
        return df

    summary = (
        df.groupby("embedding_model")
        .agg(
            avg_cosine_sim=("avg_cosine_sim", "mean"),
            avg_keyword_hit_rate=("keyword_hit_rate", "mean"),
            avg_query_time_s=("query_time_s", "mean"),
            build_time_s=("build_time_s", "first"),
        )
        .round(4)
        .sort_values(["avg_keyword_hit_rate", "avg_cosine_sim"], ascending=[False, False])
    )
    return summary


def choose_best_embedding_model(summary_df):
    if summary_df.empty:
        return list(EMBEDDING_MODELS.keys())[0]
    return summary_df.index[0]


def build_hf_llm(repo_id):
    dtype = get_torch_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        cache_dir=os.path.join(WEIGHTS_DIR, "transformers"),
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    extra_model_kwargs = {}
    if "Qwen" in repo_id:
        extra_model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        cache_dir=os.path.join(WEIGHTS_DIR, "transformers"),
        trust_remote_code=TRUST_REMOTE_CODE,
        device_map=DEVICE_MAP,
        torch_dtype=dtype,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        **extra_model_kwargs,
    )
    model.eval()

    return tokenizer, model


def get_model_main_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")


def build_messages(question, context):
    user_prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return [{"role": "user", "content": user_prompt}]


def build_prompt_text(tokenizer, question, context):
    messages = build_messages(question, context)

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return PROMPT_TEMPLATE.format(context=context, question=question)


@torch.inference_mode()
def generate_answer(tokenizer, model, question, context):
    prompt_text = build_prompt_text(tokenizer, question, context)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    model_device = get_model_main_device(model)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": MAX_TOKENS,
        "do_sample": DO_SAMPLE,
        "top_p": TOP_P if DO_SAMPLE else None,
        "top_k": TOP_K if DO_SAMPLE else None,
        "temperature": TEMPERATURE if DO_SAMPLE else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not answer:
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_text.strip()

    return answer


def benchmark_llms(chunks, embed_model_key=None):
    if embed_model_key and embed_model_key in EMBEDDING_MODELS:
        emb_name = EMBEDDING_MODELS[embed_model_key]
    else:
        embed_model_key = list(EMBEDDING_MODELS.keys())[0]
        emb_name = EMBEDDING_MODELS[embed_model_key]

    print(f"\nBuilding shared vector store with {embed_model_key}")
    embeddings = build_embedding_model(emb_name)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    rows = []

    for llm_label, cfg in LLM_MODELS.items():
        repo_id = cfg["repo_id"]

        print(f"\nLLM: {llm_label} ({repo_id})")
        print(
            f"[INFO] HF CUDA mode: device_map={DEVICE_MAP}, "
            f"use_4bit={USE_4BIT}, dtype={DTYPE_STR}"
        )

        t0 = time.time()
        try:
            tokenizer, model = build_hf_llm(repo_id)
        except Exception as e:
            print(f"[ERROR] Failed to load {llm_label}: {e}")
            rows.append(
                {
                    "llm_model": llm_label,
                    "question": "",
                    "answer": "",
                    "latency_s": 0.0,
                    "load_time_s": 0.0,
                    "model_size_mb": 0.0,
                    "error": f"load_error: {e}",
                }
            )
            clear_memory()
            continue

        load_time = time.time() - t0
        model_size_mb = safe_model_size_mb(model)
        model_device = str(get_model_main_device(model))

        print(f"Loaded in {load_time:.2f}s (~{model_size_mb:.0f} MB logical params)")
        print(f"Main model device: {model_device}")

        for tq in TEST_QUESTIONS:
            q = tq["question"]

            docs = vector_store.similarity_search(q, k=K_RETRIEVE)
            context = "\n\n".join(doc.page_content for doc in docs)

            t1 = time.time()
            try:
                answer = generate_answer(tokenizer, model, q, context)
                error = ""
            except Exception as e:
                answer = ""
                error = str(e)

            latency = time.time() - t1

            rows.append(
                {
                    "llm_model": llm_label,
                    "question": q,
                    "answer": answer[:500],
                    "latency_s": round(latency, 2),
                    "load_time_s": round(load_time, 2),
                    "model_size_mb": round(model_size_mb, 1),
                    "error": error,
                }
            )

            short_q = q[:50]
            short_a = (answer[:75] + "...") if len(answer) > 75 else answer
            print(f"    Q: {short_q:<52} ({latency:.2f}s)")
            print(f"    A: {short_a if short_a else '[ERROR / EMPTY ANSWER]'}")

        del model
        del tokenizer
        clear_memory()

    del vector_store
    del embeddings
    clear_memory()

    return pd.DataFrame(rows)


def summarise_llms(df):
    if df.empty:
        return df

    summary = (
        df.groupby("llm_model")
        .agg(
            avg_latency_s=("latency_s", "mean"),
            model_size_mb=("model_size_mb", "first"),
            load_time_s=("load_time_s", "first"),
            questions_answered=(
                "answer",
                lambda s: sum(1 for a in s if isinstance(a, str) and a.strip()),
            ),
            errors=(
                "error",
                lambda s: sum(1 for e in s if isinstance(e, str) and e.strip()),
            ),
        )
        .round(2)
        .sort_values("avg_latency_s")
    )
    return summary


def main():
    args = [a.lower() for a in sys.argv[1:]]

    if not args or args[0] in ("-h", "--help"):
        print("Usage:")
        print("  python benchmark.py embeddings")
        print("  python benchmark.py llms")
        print("  python benchmark.py all")
        print()
        print("Environment variables:")
        print("  USE_CUDA=1")
        print("  USE_4BIT=1")
        print("  DTYPE=float16")
        print("  DEVICE_MAP=auto")
        print("  EMBED_DEVICE=cuda")
        print("  MAX_TOKENS=256")
        print("  MAX_INPUT_TOKENS=1800")
        print("  DO_SAMPLE=0")
        print("  TEMPERATURE=0.1")
        print("  TOP_P=0.95")
        print("  TOP_K=50")
        print("  EMBED_BATCH_SIZE=32")
        print("  TRUST_REMOTE_CODE=1")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "hub"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "transformers"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "sentence_transformers"), exist_ok=True)

    print_env_summary()

    print(f"Extracting text from PDFs in {PDF_FOLDER}")
    raw_text = extract_all_pdfs(PDF_FOLDER)
    if not raw_text.strip():
        print("No text extracted. Add PDFs to pdfs/ and retry.")
        return

    chunks = chunk_text(raw_text)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    run_embed = args[0] in ("embeddings", "all")
    run_llms = args[0] in ("llms", "all")

    best_embedding_key = None

    if run_embed:
        print("Embedding Model Comparison:")
        embed_df = benchmark_embeddings(chunks)

        embed_path = os.path.join(RESULTS_DIR, "embedding_results.csv")
        embed_df.to_csv(embed_path, index=False)
        print(f"\nEmbedding results saved to {embed_path}")

        embed_summary = summarise_embeddings(embed_df)
        print("\nEmbedding Summary")
        print(embed_summary.to_string())

        summary_path = os.path.join(RESULTS_DIR, "embedding_summary.csv")
        embed_summary.to_csv(summary_path)

        best_embedding_key = choose_best_embedding_model(embed_summary)
        print(f"\nBest embedding model selected for LLM benchmark: {best_embedding_key}")

    if run_llms:
        if best_embedding_key is None:
            best_embedding_key = list(EMBEDDING_MODELS.keys())[0]

        print("Local LLM Comparisons:")
        llm_df = benchmark_llms(chunks, best_embedding_key)

        if llm_df.empty:
            print("\nNo LLM results were generated.")
        else:
            llm_path = os.path.join(RESULTS_DIR, "llm_results.csv")
            llm_df.to_csv(llm_path, index=False)
            print(f"\nLLM results saved to {llm_path}")

            llm_summary = summarise_llms(llm_df)
            print("\nLLM Summary")
            print(llm_summary.to_string())

            summary_path = os.path.join(RESULTS_DIR, "llm_summary.csv")
            llm_summary.to_csv(summary_path)


if __name__ == "__main__":
    main()
