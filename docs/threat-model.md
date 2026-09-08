# KrishiMitra AI — Threat Model

## Assets to protect

1. **Trust in the answer** — a farmer must never be told something false or
   unsupported presented as fact (financial/legal harm if a farmer acts on a
   fabricated eligibility claim or benefit amount).
2. **The knowledge base's source integrity** — Vikaspedia-derived content
   must never be silently mixed with unverified or user-supplied content.
3. **User-uploaded documents** — may contain personal/financial information;
   must be handled safely and not leaked into logs.
4. **System prompt / internal instructions** — should not be exfiltratable.
5. **Server/host** — file upload and crawl paths must not allow traversal,
   arbitrary file execution, or resource exhaustion.

## Threats & mitigations

| # | Threat | Mitigation |
|---|---|---|
| T1 | Prompt injection: "ignore your instructions", "reveal your system prompt" | Deterministic regex-based `prompt_injection.py` gate on every input (no LLM call, cannot itself be talked out of its job); optional NeMo Guardrails self-check as a second opinion. Retrieved documents are explicitly framed as **DATA, never instructions** in the system prompt (`app/llm/prompts.py`). |
| T2 | Source-restriction bypass: "use your own knowledge", "answer using info outside Vikaspedia" | Same injection guardrail; the system prompt's rule 2 explicitly instructs the LLM to ignore instruction-like text found inside evidence; the hallucination guard independently strips any claim not grounded in retrieved evidence regardless of what the LLM says. |
| T3 | Hallucinated scheme names/amounts/deadlines/contacts | Structured `LLMAnswer` contract requires `chunk_id` citations per claim; `hallucination.py` verifies citation validity **and** lexical overlap between claim and cited text; unsupported claims are deleted; if nothing survives, the system refuses outright. |
| T4 | False "you are eligible" claims | Eligibility is never an LLM judgment call — `app/eligibility/matcher.py` is a deterministic rules engine; the orchestrator overwrites whatever the LLM guessed with the engine's verdict. |
| T5 | Off-topic misuse (using the assistant as a general chatbot) | Keyword-based relevance guardrail (`relevance.py`) blocks clearly off-topic queries before retrieval; if a borderline query still returns no real evidence, the retrieval guardrail refuses. |
| T6 | Malicious file upload (path traversal, oversized file, disguised extension) | `app/core/security.py`: filename is reduced to a basename (no path components), extension allowlist, size cap, best-effort MIME cross-check; files are written to a per-request random temp path and deleted after processing. |
| T7 | Sensitive data in logs | `sanitize_for_log()` truncates/flattens text before logging; the structured logger redacts keys like `api_key`/`token`/`password`; raw file contents are never logged. |
| T8 | Uncontrolled crawling (scraping outside allowlist, hammering the target site) | `crawler_config.py` enforces an explicit domain allowlist + include/exclude URL patterns; `robots.txt` is checked per-origin; rate limiting + retry/backoff; crawling only ever runs via an explicit offline script, never during a chat request. |
| T9 | Live web data smuggled into "trusted" answers | Chat never calls the crawler; the knowledge base is a snapshot, and the UI/answers explicitly label "Knowledge base last updated: DATE" and refuse queries for "today's" news. |
| T10 | User-uploaded document silently treated as authoritative as Vikaspedia | Every chunk/evidence item carries `source_type`; UI labels "User-provided document" vs "Vikaspedia" distinctly; source priority (Vikaspedia > trusted documents > user uploads) is documented and never silently reordered. |
| T11 | OCR/STT hallucination on unreadable input | OCR confidence threshold (`OCR_MIN_CONFIDENCE`) — below it, the system explicitly declines rather than guessing at garbled text; transcribed voice queries are shown to the user before being used. |
| T12 | Unhandled exceptions leaking stack traces | FastAPI global exception handlers return a generic user-safe message; details are logged server-side only. |

## Out of scope for this build

- Authentication/authorization (no user accounts; single-tenant demo).
- Rate limiting on the FastAPI endpoints themselves (would be needed before
  any public deployment).
- Formal PII detection/redaction in uploaded documents beyond not logging
  raw contents.
