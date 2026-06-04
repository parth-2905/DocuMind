import os
from groq import Groq

# ── Thresholds ────────────────────────────────────────────────────────────────
FULL_CONTEXT_THRESHOLD = 2.0   # above this → answer purely from document
PARTIAL_THRESHOLD      = 0.0   # between 0 and 2 → hybrid answer
                                # below 0 → full fallback

def get_llm():
    return Groq(api_key=os.environ["GROQ_API_KEY"])

def _deduplicate(chunks: list, threshold: int = 100) -> list:
    seen = []
    unique = []
    for chunk in chunks:
        text = chunk["text"].strip()
        is_duplicate = any(
            len(set(text.split()) & set(s.split())) / max(len(text.split()), 1) > 0.8
            for s in seen
        )
        if not is_duplicate:
            unique.append(chunk)
            seen.append(text)
    return unique

def _determine_source_type(top_score):
    """Determine answer source based on reranker confidence score."""
    if top_score is None:
        return "document"
    if top_score >= FULL_CONTEXT_THRESHOLD:
        return "document"
    elif top_score >= PARTIAL_THRESHOLD:
        return "partial"
    else:
        return "fallback"

def generate_response(query, chunks, llm, notes_mode="medium", top_reranker_score=None):
    chunks = _deduplicate(chunks)

    source_type = _determine_source_type(top_reranker_score)

    context_parts = []
    for c in chunks:
        prefix = f"[Page {c['page']}]"
        if c.get("low_quality"):
            prefix += " [low OCR confidence — treat cautiously]"
        context_parts.append(f"{prefix}: {c['text']}")
    context = "\n\n".join(context_parts)

    mode_aliases = {
        "qa": "medium",
        "qa_brief": "brief",
        "qa_detailed": "detailed",
    }
    resolved_mode = mode_aliases.get(notes_mode, notes_mode)

    from src.notes import get_prompt
    prompt = get_prompt(resolved_mode, context, query)

    max_tokens_map = {
        "brief": 1024,
        "medium": 2048,
        "detailed": 3000,
    }
    max_tokens = max_tokens_map.get(resolved_mode, 1024)

    # ── System prompt based on source type ───────────────────────────────────
    if source_type == "document":
        system_prompt = """You are a strict study assistant helping college students.
ONLY answer using information explicitly present in the provided context.
Write detailed, thorough notes that fully cover every concept mentioned in the context.
Always cite the source page number using the format (p. X) at the end of each section or key point.
If the topic is not covered in the context at all, respond with ONLY: 'I could not find this in the document.'
Do NOT use any knowledge outside the provided context under any circumstances."""

    elif source_type == "partial":
        system_prompt = """You are a helpful study assistant. The document contains limited information on this topic.
Structure your response in two clearly labeled parts:

📄 From your document:
Answer using ONLY the information present in the provided context. Cite page numbers using (p. X).
If context is too limited to answer properly, say: 'The document only briefly mentions this.'

🌐 From general knowledge:
Supplement with your own knowledge to give a complete answer.
Clearly distinguish what comes from the document vs general knowledge."""

    else:  # fallback
        system_prompt = """You are a helpful study assistant. The uploaded document does not contain relevant information for this query.
Answer using your general knowledge. Be clear and concise.
Start your response with exactly: '⚠️ Not found in document — answering from general knowledge:\n\n'
Then provide your answer."""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=max_tokens
    )

    answer = response.choices[0].message.content
    sources = [{"page": c["page"], "text": c["text"][:200]} for c in chunks]

    return {
        "answer": answer,
        "sources": sources,
        "mode": resolved_mode,
        "source_type": source_type  # "document" | "partial" | "fallback"
    }
