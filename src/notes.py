from typing import Literal

NotesMode = Literal["brief", "medium", "detailed"]

NOTES_PROMPTS = {
    "brief": """You are a study assistant. First check if the topic/question is covered in the context below.
If the topic is NOT mentioned anywhere in the context, respond with ONLY: 'I could not find this in the document.'
If it IS covered, generate BRIEF notes following these rules:
- 6 to 8 bullet points
- Each bullet MUST be 2-3 sentences minimum
- For each bullet: state the concept, explain how it works, and give context or an example from the document
- Cover: definition, core mechanism, key components, example, and at least one application or advantage
- Include any specific numbers, formulas, or technical terms from the context
- Never write one-liners — every bullet must teach something complete

Context: {context}
Topic/Question: {question}
Brief Notes:""",

    "medium": """You are a study assistant. First check if the topic/question is covered in the context below.
If the topic is NOT mentioned anywhere in the context, respond with ONLY: 'I could not find this in the document.'
If it IS covered, generate MEDIUM notes following these rules:
- 5 to 7 sections with bold headings
- Each section MUST have a minimum of 4-6 sentences
- Cover ALL of the following that appear in the context:
  * Full definition with explanation
  * Step-by-step working mechanism
  * Key components and their roles
  * Concrete examples with details from the document
  * Advantages and disadvantages
  * Technical terms defined inline
  * Connections to related concepts
- Write as if explaining to a student answering a 10-mark exam question
- Do not skip any concept present in the context
- Use specific numbers, register names, formulas, and table references from the context

Context: {context}
Topic/Question: {question}
Medium Notes:""",

    "detailed": """You are a study assistant. First check if the topic/question is covered in the context below.
If the topic is NOT mentioned anywhere in the context, respond with ONLY: 'I could not find this in the document.'
If it IS covered, generate DETAILED notes following these rules:
- Minimum 800 words
- Bold headings and subheadings throughout
- For every concept cover: full definition, intuition, step-by-step mechanics, all examples with complete details, every technical term explained, advantages, disadvantages, edge cases, and connections to other concepts
- Explain every equation, register operation, and table reference term by term
- Write full paragraphs under each heading — not just bullet points
- Be exhaustive — treat this as the only study material the student has
- Use all specific details, numbers, register names, clock cycles, and examples from the context

Context: {context}
Topic/Question: {question}
Detailed Notes:""",
}

def get_prompt(mode: NotesMode, context: str, question: str) -> str:
    template = NOTES_PROMPTS.get(mode, NOTES_PROMPTS["medium"])
    return template.format(context=context, question=question)

def get_mode_label(mode: NotesMode) -> str:
    return {
        "brief":    "Brief - key points only",
        "medium":   "Medium - balanced summary",
        "detailed": "Detailed - comprehensive notes",
    }.get(mode, mode)
