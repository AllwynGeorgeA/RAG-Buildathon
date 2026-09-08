# KrishiMitra AI — 3-Minute Demo Script

**Setup beforehand (do this before judges arrive, not live):**
```bash
python scripts/demo_seed.py          # only if data/vector_store is empty
uvicorn app.api.main:app --port 8000 # serves both the API and the green UI
streamlit run app/ui/streamlit_app.py # optional — classic UI, has a link back to the green one
```
Open **`http://localhost:8000/ui/chatbot-ui-green.html`** — this is the
primary demo surface. If you need a public link instead of screen-share,
`npx localtunnel --port 8000` from a spare terminal.

**Know before you demo — which screens are real:**
- ✅ **Chat** is fully live: real conversations, real API calls, real answers.
- ⚠️ **Dashboard / Insights / Eligibility Checker / Documents / Schemes
  Library** (the other sidebar links) are **static design mockups** with
  placeholder numbers — don't click into them expecting live data; if you
  do by accident, say so plainly rather than let it look like a bug.
- Real document upload / OCR / voice transcription only exist in the
  **Streamlit** app's "📎 Upload documents" tab, not the green UI yet.

**Know before you demo — what's actually indexed:** the knowledge base
covers **PM-KISAN, PMFBY, Soil Health Card, Kisan Credit Card, PMKSY,
Interest Subvention, Agricultural/Crop Insurance, and PM-KUSUM** in depth.
Stick to these by name for guaranteed strong answers. A few other scheme
names (NREGA, eNAM, PM Kisan Maandhan, PMJJBY/PMSBY) appear only as passing
mentions inside those pages — asking about them directly gets an honest
refusal, not a guess (see step 7 — this is a feature to show off, not a gap
to avoid, if you frame it deliberately).

---

**1. Open the app** on the hero screen. Point out the green "AI Chatbot"
branding, the sidebar's real conversation history (empty or with prior
chats grouped Today/Yesterday/Previous 7 Days), and the three suggestion
cards — these are real questions, not filler.

**2. Click a suggestion card** (or ask): *"Am I eligible for PM-KISAN with
2 acres in Uttar Pradesh?"*
Show: the confidence badge (HIGH/MEDIUM/LOW), the eligibility verdict badge
on the scheme card (`likely_eligible` / `possibly_eligible` /
`insufficient_information` / `likely_not_eligible`) — never a plain LLM
guess — the "missing info" line, and "View N sources" expanding to the
real Vikaspedia section + link.

**3. Click a follow-up chip** (e.g. "What crop do you grow?"). Point out it
prompts *you* to type the answer rather than re-asking itself — a subtle
but real correctness detail (a first version of this looped forever
resending its own question; worth mentioning if asked about engineering
rigor).

**4. Ask the objection:** *"My neighbor says this scheme is only for large
farmers. Is that true?"*
Show it addresses only that claim from evidence — no arguing, no
speculation, and says so plainly if the source doesn't confirm it.

**5. Toggle the 🌐 web search icon** in the composer, then ask something
like *"PM-KISAN official website"*. Show the answer is unchanged/cited as
before, but a **separate, dashed "⚠️ unverified" box** appears below it
with live results — make the point explicit: this is opt-in, off by
default, and never blended into the cited answer above it.

**6. Ask something clearly off-topic:** *"What is today's weather
forecast?"* Show the relevance guardrail's refusal.

**7. Ask about a scheme name that's only mentioned in passing:**
*"Tell me about PM Kisan Maandhan Yojana pension scheme."* Show the honest
"the knowledge base does not have information about..." refusal — frame
this as the point of the whole project: **it would rather admit it doesn't
know than invent a benefit amount or eligibility rule.**

**8. Rename or pin a conversation** in the sidebar (double-click a title,
or hover to pin/delete) to show chat history is genuinely persisted
server-side (SQLite), not a client-side illusion — refresh the page to
prove it survives.

**9. (Optional, if time allows) Switch to the Streamlit app**, open
"📎 Upload documents", upload a PDF or photo of a scheme notice, and ask a
question about it — shows the multimodal ingestion pipeline and the
**"User-provided document"** label that keeps it distinct from Vikaspedia
content.

**10. Run the evaluation harness** in a terminal: `python scripts/evaluate.py`
and show the printed metrics — Retrieval Precision, Groundedness, Citation
Accuracy, Hallucination Rate, Refusal Accuracy. `pytest` (81+ passing) if
someone asks about testing.

---

**Closing line:** "Every claim you saw was traceable to a real Vikaspedia
section, and every time it didn't know something, it said so instead of
guessing. That's the product — not a chatbot that sounds confident, an
assistant that's *actually checkable*."
