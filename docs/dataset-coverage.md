# What's actually in the demo knowledge base

Ground-truth reference for live Q&A / judging — checked against the real
running app on 2026-09-08, not just inferred from file names. Use this so
no live question catches you off guard.

## Well-covered — safe to name directly

These 8 schemes have real, multi-section Vikaspedia content indexed and
reliably produce HIGH-confidence, cited, eligibility-scored answers:

- **PM-KISAN** (Pradhan Mantri Kisan Samman Nidhi)
- **PMFBY** (Pradhan Mantri Fasal Bima Yojana / crop insurance)
- **Soil Health Card**
- **Kisan Credit Card (KCC)**
- **PMKSY** (ground-water irrigation / micro-irrigation)
- **Interest Subvention Scheme**
- **Agricultural / Crop Insurance** (general, complementing PMFBY)
- **PM-KUSUM** (solar irrigation pump subsidy)

Any farmer profile detail (state, crop, landholding, farmer type) you throw
at a question about *these* schemes will be handled reasonably well: entity
extraction runs against a large built-in state/crop vocabulary at query
time, not just the handful of names that happen to be graph nodes — so
e.g. "Uttar Pradesh" or "wheat" work fine in a query even though they
aren't graph nodes themselves (graph-confirmation is a bonus score boost
for the few combinations that were extracted from the source text, not a
requirement for a good answer).

## Mentioned only in passing — will honestly refuse, don't lead with these

These names surface in the knowledge graph because another scheme's page
mentions them in a list of "related schemes" — they don't have their own
indexed content, so asking about them directly gets a genuine "I don't have
that information" refusal, not a hallucinated answer:

- NREGA / MGNREGA Job Card
- eNAM
- PM Kisan Maandhan Yojana (pension)
- Pradhan Mantri Jeevan Jyoti Bima / Pradhan Mantri Suraksha Bima
- National / Revenue / Weather-Based / Unified Package crop insurance variants
  (beyond the main PMFBY content)

**Turn this into a strength, not a risk**: asking about one of these live
and getting an honest refusal instead of a made-up answer is a *better*
demo moment than avoiding the topic — see step 7 of `demo-script.md`.

## Tested sample questions (verified against the live app)

Guaranteed strong answers:
- *"Am I eligible for PM-KISAN with 2 acres in Uttar Pradesh?"* → HIGH/MEDIUM
  confidence, `likely_eligible`, 2+ real citations.
- *"What is PM-KISAN?"* / *"How do I apply for the Kisan Credit Card?"*

Guaranteed honest refusal (use deliberately, not by accident):
- *"Tell me about the PM Kisan Maandhan Yojana pension scheme."*
- *"What is the NREGA Job Card scheme?"*

## Regenerating this reference

If you re-crawl or re-seed the knowledge base, regenerate the entity list
with:
```bash
python -c "
from app.graph.knowledge_graph import get_knowledge_graph
kg = get_knowledge_graph()
for n in kg.nodes_of_type('Scheme'):
    print(kg.graph.nodes[n].get('name', n))
"
```
