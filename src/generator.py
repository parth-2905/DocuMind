import os
from groq import Groq

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

def generate_response(query, chunks, llm, notes_mode="medium"):
    chunks = _deduplicate(chunks)

    context_parts = []
    for c in chunks:
        prefix = f"[Page {c['page']}]"
        if c.get("low_quality"):
            prefix += " [low OCR confidence — treat cautiously]"
        context_parts.append(f"{prefix}: {c['text']}")
    context = "\n\n".join(context_parts)
    
     mode_aliases = {
        "qa":         "medium",
        "qa_brief":   "brief",
        "qa_detailed": "detailed",
    }
    resolved_mode = mode_aliases.get(notes_mode, notes_mode)

    from src.notes import get_prompt
    prompt = get_prompt(notes_mode, context, query)

    max_tokens_map = {
    "brief": 1024,
    "medium": 2048,
    "detailed": 3000,
    }
    max_tokens = max_tokens_map.get(notes_mode, 1024)
    if notes_mode in ["brief", "medium", "detailed"]:
        system_prompt = """You are a strict study assistant helping college students.
ONLY answer using information explicitly present in the provided context.
Write detailed, thorough notes that fully cover every concept mentioned in the context.
The more content in the context, the more detailed your notes should be.
Always cite the source page number using the format (p. X) at the end of each section or key point.
If the topic is not covered in the context at all, respond with ONLY: 'I could not find this in the document.'
Do NOT use any knowledge outside the provided context under any circumstances."""
    else:
        system_prompt = """You are a strict document Q&A assistant. You have ONE absolute rule:

If the exact answer to the question cannot be found explicitly stated in the provided context, you MUST respond with ONLY this sentence:
'I could not find this in the document.'

DO NOT infer, guess, or reason about topics not explicitly in the context.
DO NOT use any outside knowledge whatsoever.
DO NOT generate explanations or related content if the topic is not directly addressed in the context.
DO NOT respond with anything other than 'I could not find this in the document.' when the answer is not present.

The context is your ONLY source of truth. Nothing else exists."""

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
    return {"answer": answer, "sources": sources, "mode": notes_mode}
