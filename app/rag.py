# app/rag.py
import os
import json
import sqlite3
import math
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "vectors.db")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_query(query: str) -> List[float]:
    resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=[query])
    return resp.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_documents(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    print("Embed query...")
    query_embedding = embed_query(query)
    print("Query embedded. Loading docs from SQLite...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT text, source_file, chunk_index, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    print(f"Loaded {len(rows)} docs. Computing similarities...")

    scored = []
    for text, source_file, chunk_index, emb_json in rows:
        if filters and "source_file" in filters:
            if source_file != filters["source_file"]:
                continue
        emb = json.loads(emb_json)
        score = cosine_similarity(query_embedding, emb)
        scored.append((score, text, source_file, chunk_index))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    print("Done retrieving.")
    return [
        {"text": text, "metadata": {"source_file": sf, "chunk_index": ci}}
        for _, text, sf, ci in top
    ]


SYSTEM_PROMPT = """
Ти си старши експерт по Business Continuity и GRC в голяма организация.
Работиш по стандарти като ISO 22301 и вътрешните BCP политики.
Основната ти задача е да помагаш на GRC анализатора да проектира, поддържа и подобрява Business Continuity програмата.

Правила:
- Използвай предоставения контекст (документни откъси) като основен източник.
- Ако контекстът не стига, кажи ясно каква допълнителна информация е нужна.
- Не измисляй регулации или политики; ако нещо е предположение, го маркирай.
- Форматирай отговорите структурирано: кратко резюме, после стъпки, после рискове/гепове.
""".strip()


def build_prompt(user_query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_texts = []
    for c in contexts:
        meta = c.get("metadata", {})
        src = meta.get("source_file", "unknown")
        idx = meta.get("chunk_index", "?")
        context_texts.append(f"[{src} / chunk {idx}]\n{c['text']}")

    full_context = "\n\n---\n\n".join(context_texts) if context_texts else "Няма наличен контекст."

    user_content = f"""
Контекст от документи:
{full_context}

---

Въпрос на GRC анализатора:
{user_query}

Отговори като BCP/GRC експерт, структурирано и ясно.
""".strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def call_llm(messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY не е настроен в .env."
    try:
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"OpenAI грешка: {e}")
        return f"Грешка: {e}"


def ask_bcp_assistant(
    user_query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    print("1. Retrieve documents...")
    contexts = retrieve_documents(user_query, top_k=top_k, filters=filters)
    print("2. Build prompt...")
    messages = build_prompt(user_query, contexts)
    print("3. Call LLM...")
    answer = call_llm(messages)
    print("4. Done!")
    return {"answer": answer, "contexts": contexts}