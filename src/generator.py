import os
from groq import Groq

def get_llm():
    return Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_response(query: str, chunks: list, llm) -> dict:
    context = "\n\n".join([
        f"[Page {c['page']}]: {c['text']}" for c in chunks
    ])
    prompt = f"""You are a helpful study assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say 'I could not find this in the document.'

Context:
{context}

Question: {query}

Answer:"""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512
    )
    answer = response.choices[0].message.content
    sources = [{"page": c["page"], "text": c["text"][:200]} for c in chunks]
    return {"answer": answer, "sources": sources}
