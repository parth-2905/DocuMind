import os
from groq import Groq

def get_llm():
    return Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_response(query, chunks, llm, notes_mode="medium"):
    context = '\n\n'.join([f"[Page {c['page']}]: {c['text']}" for c in chunks])
    from src_notes import get_prompt
    prompt = get_prompt(notes_mode, context, query)
    
    max_tokens = 512 if notes_mode == "brief" else 1024 if notes_mode == "medium" else 2048
    
    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a strict study assistant with one absolute rule:
ONLY answer using information explicitly present in the provided context.
If the question is not answerable from the context, respond with ONLY this exact sentence: 'I could not find this in the document.'
Do NOT add anything after that sentence.
Do NOT generate notes or additional content if the question is irrelevant.
Do NOT use any knowledge outside the provided context under any circumstances.
Violating this rule is not acceptable."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=max_tokens
    )
    answer = response.choices[0].message.content
    sources = [{"page": c["page"], "text": c["text"][:200]} for c in chunks]
    return {"answer": answer, "sources": sources, "mode": notes_mode}
