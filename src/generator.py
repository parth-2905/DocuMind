import os
from groq import Groq

def get_llm():
    return Groq(api_key=os.environ["GROQ_API_KEY"])

def _deduplicate(chunks: list, threshold: int = 100) -> list:
    """Remove chunks with heavily overlapping text."""
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

def generate_response(query, chunks, llm, notes_mode="medium"):
    chunks = _deduplicate(chunks)

    context_parts = []
    for c in chunks:
        prefix = f"[Page {c['page']}]"
        if c.get("low_quality"):
            prefix += " [low OCR confidence — treat cautiously]"
        context_parts.append(f"{prefix}: {c['text']}")
    context = "\n\n".join(context_parts)

    from src.notes import get_prompt
    prompt = get_prompt(notes_mode, context, query)

    max_tokens = 1024 if notes_mode == "brief" else 2048 if notes_mode == "medium" else 3000

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a strict study assistant helping college students.
ONLY answer using information explicitly present in the provided context.
Write detailed, thorough notes that fully cover every concept mentioned in the context.
The more content in the context, the more detailed your notes should be.
Always cite the source page number using the format (p. X) at the end of each section or key point.
If the question is not answerable from the context, respond with ONLY: 'I could not find this in the document.'
Do NOT use any knowledge outside the provided context under any circumstances."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=max_tokens
    )

    answer = response.choices[0].message.content
    sources = [{"page": c["page"], "text": c["text"][:200]} for c in chunks]
    return {"answer": answer, "sources": sources, "mode": notes_mode}
