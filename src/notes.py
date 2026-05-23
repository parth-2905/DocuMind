from typing import Literal

NotesMode = Literal["brief", "medium", "detailed"]

NOTES_PROMPTS = {
    "brief": """You are a concise study assistant. Using ONLY the context below, generate BRIEF notes.
Rules:
- Maximum 5 bullet points
- Each bullet under 15 words
- Only the most critical concepts
- No examples, no elaboration
- Start each bullet with a dash (-)

Context: {context}
Topic/Question: {question}
Brief Notes:""",

    "medium": """You are a helpful study assistant. Using ONLY the context below, generate MEDIUM-LEVEL notes.
Rules:
- 3 to 5 sections with short bold headings
- 2 to 3 sentences per section
- Include important definitions
- One example per section where relevant

Context: {context}
Topic/Question: {question}
Medium Notes:""",

    "detailed": """You are a thorough study assistant. Using ONLY the context below, generate HIGHLY DETAILED notes.
Rules:
- Comprehensive section breakdown with bold headings
- Full explanations with reasoning, not just facts
- Multiple examples from the context
- Define all key terms
- Show connections between concepts
- Cover edge cases and nuances mentioned in the context

Context: {context}
Topic/Question: {question}
Detailed Notes:"""
}

def get_prompt(mode: NotesMode, context: str, question: str) -> str:
    template = NOTES_PROMPTS.get(mode, NOTES_PROMPTS["medium"])
    return template.format(context=context, question=question)

def get_mode_label(mode: NotesMode) -> str:
    return {
        "brief": "Brief - key points only",
        "medium": "Medium - balanced summary",
        "detailed": "Detailed - comprehensive notes"
    }.get(mode, mode)
