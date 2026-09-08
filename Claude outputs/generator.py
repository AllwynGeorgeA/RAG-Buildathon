"""
Answer generator — builds a prompt from retrieved chunks and calls the LLM.
Returns a structured answer with a scheme info card when applicable.
"""
from __future__ import annotations
import os
import json
from openai import OpenAI

_client: OpenAI | None = None

FEW_SHOT = """---
Q: What is PM-KISAN?
A: PM-KISAN (Pradhan Mantri Kisan Samman Nidhi) provides ₹6,000/year in three equal instalments of ₹2,000 directly to eligible farmer families' bank accounts. Eligibility: small and marginal farmers with combined landholding up to 2 hectares. Apply via the PM-KISAN portal or nearest CSC centre.
---
Q: How can a farmer get crop insurance under PMFBY?
A: Under PMFBY (Pradhan Mantri Fasal Bima Yojana), farmers pay a low premium (2% for kharif, 1.5% for rabi, 5% for horticulture). Enrol through your bank before the cut-off date for each season. Claims are settled within two months of crop loss assessment. Contact your district agriculture officer or Kisan Call Centre 1551 for help.
---"""

SYSTEM_PROMPT = f"""You are KisanBot, an expert AI assistant for Indian farmers.
Your job: answer questions about government agricultural schemes, subsidies, loans,
and farming practices clearly and accurately, using the provided context.

Rules:
1. Base your answer primarily on the CONTEXT provided.
2. If context is insufficient, say so honestly — do not hallucinate scheme details.
3. Use simple language suitable for farmers; include key numbers (amounts, deadlines, eligibility).
4. Always mention how to apply or who to contact when relevant.
5. End every answer with a ★ SCHEME QUICK CARD in JSON if you identified a specific scheme.

Few-shot examples:
{FEW_SHOT}
"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def generate(
    query: str,
    chunks: list[str],
    pdf_chunks: list[str] | None = None,
    chat_history: list[dict] | None = None,
) -> tuple[str, dict | None]:
    """
    Generate an answer from retrieved chunks.

    Returns:
        (answer_text, scheme_card_dict_or_None)
    """
    context_parts = []
    if chunks:
        context_parts.append("=== Knowledge Base ===\n" + "\n---\n".join(chunks))
    if pdf_chunks:
        context_parts.append("=== Uploaded Document ===\n" + "\n---\n".join(pdf_chunks))

    context = "\n\n".join(context_parts) if context_parts else "No context available."

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for turn in chat_history[-6:]:  # last 3 exchanges
            messages.append(turn)

    messages.append(
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {query}\n\n"
                "Answer the question using the context. "
                "If you identify a specific scheme, append a ★ SCHEME QUICK CARD "
                'as a JSON code block: ```json\n{"scheme": "...", "benefit": "...", '
                '"eligibility": "...", "how_to_apply": "...", "helpline": "..."}\n```'
            ),
        }
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
    )
    full_text: str = resp.choices[0].message.content.strip()

    # Extract scheme card JSON if present
    scheme_card: dict | None = None
    if "```json" in full_text:
        try:
            json_start = full_text.index("```json") + 7
            json_end = full_text.index("```", json_start)
            scheme_card = json.loads(full_text[json_start:json_end].strip())
            # Remove the JSON block from display text
            answer_text = full_text[:full_text.index("```json")].strip()
        except Exception:
            answer_text = full_text
    else:
        answer_text = full_text

    return answer_text, scheme_card


def evaluate_answer(query: str, answer: str, chunks: list[str]) -> str:
    """
    Rate the answer quality A-F based on evidence alignment.
    Returns a one-letter grade.
    """
    context = "\n---\n".join(chunks[:3]) if chunks else "none"
    prompt = (
        f"Rate the following answer to a farmer's question on a scale A (excellent) "
        f"to F (poor/hallucinated).\n\n"
        f"Question: {query}\n\nContext used:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Reply with a single letter grade (A, B, C, D, or F) followed by a one-line reason."
    )
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "N/A"
