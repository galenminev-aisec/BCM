# app/ingest.py
import os
import json
import sqlite3
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DB_PATH = os.path.join(BASE_DIR, "vectors.db")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
client = OpenAI(api_key=OPENAI_API_KEY)


def load_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        texts.append(t)
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start = end - overlap
    return chunks


def embed_texts(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
    return all_embeddings


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            source_file TEXT,
            chunk_index INTEGER,
            embedding TEXT
        )
    """)
    conn.commit()
    return conn


def ingest():
    files = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for fname in filenames:
            if fname.lower().endswith(".pdf"):
                files.append(os.path.join(root, fname))

    if not files:
        print(f"Няма .pdf файлове в {DATA_DIR}.")
        return

    texts = []
    metadatas = []

    for fpath in files:
        print(f"Индексирам: {fpath}")
        raw_text = load_text_from_pdf(fpath)
        if not raw_text.strip():
            print(f"⚠ Няма текст от: {fpath}")
            continue
        chunks = chunk_text(raw_text)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"source_file": os.path.basename(fpath), "chunk_index": i})

    if not texts:
        print("Няма текст за индексиране.")
        return

    print(f"Правя embeddings за {len(texts)} chunk-а...")
    embeddings = embed_texts(texts)

    conn = init_db()
    conn.execute("DELETE FROM documents")

    for text, meta, emb in zip(texts, metadatas, embeddings):
        conn.execute(
            "INSERT INTO documents (text, source_file, chunk_index, embedding) VALUES (?, ?, ?, ?)",
            (text, meta["source_file"], meta["chunk_index"], json.dumps(emb))
        )

    conn.commit()
    conn.close()
    print(f"Готово. Индексирани са {len(texts)} chunk-а в SQLite.")


if __name__ == "__main__":
    ingest()