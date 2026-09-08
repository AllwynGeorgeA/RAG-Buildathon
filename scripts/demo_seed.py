#!/usr/bin/env python
"""
Seeds a deterministic "Demo knowledge base" so the Buildathon demo works
even with no internet access at demo time (spec section 35).

Primary path: ingest data/raw/vikaspedia_documents.json — the output of a
real `scripts/crawl_vikaspedia.py` run against live Vikaspedia (this is what
ships in the repo; it was produced by an actual crawl, not invented).

Fallback path: if that file is missing, ingest a small embedded set of
excerpts that were themselves copied verbatim from real Vikaspedia pages
during development (with their real source URLs preserved) — so even the
fallback never fabricates scheme content, it's just a smaller, hardcoded
slice of the same real material.

Usage:
    python scripts/demo_seed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import get_logger  # noqa: E402
from app.ingestion.pipeline import ingest_documents  # noqa: E402
from app.rag.metadata import RawDocument  # noqa: E402

logger = get_logger(__name__)

PRIMARY_DATASET = Path("data/raw/vikaspedia_documents.json")

# Real excerpts from https://en.vikaspedia.in (fetched via the Playwright
# crawler during development). Used ONLY if PRIMARY_DATASET is unavailable.
_FALLBACK_DOCS = [
    dict(
        title="Pradhan Mantri KISAN Samman Nidhi | Vikaspedia - English - Agriculture",
        section="Objective",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/policies-and-schemes/crops-related/pradhan-mantri-kisan-samman-nidhi",
        content=(
            "With a view to provide income support to all land holding eligible farmer families, "
            "the Government has launched PM-KISAN. The scheme aims to supplement the financial "
            "needs of the farmers in procuring various inputs to ensure proper crop health and "
            "appropriate yields, commensurate with the anticipated farm income."
        ),
    ),
    dict(
        title="Pradhan Mantri KISAN Samman Nidhi | Vikaspedia - English - Agriculture",
        section="Benefits and Eligibility conditions",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/policies-and-schemes/crops-related/pradhan-mantri-kisan-samman-nidhi",
        content=(
            "Under this scheme, every eligible farming family receives an annual benefit of Rs 6,000, "
            "distributed in three equal installments of Rs 2,000 every four months. This amount is "
            "directly transferred to the bank accounts of the beneficiaries. Small and marginal farmer "
            "families with combined landholding up to 2 hectares are eligible."
        ),
    ),
    dict(
        title="Pradhan Mantri KISAN Samman Nidhi | Vikaspedia - English - Agriculture",
        section="Exclusion Categories",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/policies-and-schemes/crops-related/pradhan-mantri-kisan-samman-nidhi",
        content=(
            "The following categories of beneficiaries of higher economic status shall not be eligible "
            "for benefit under the scheme: All Institutional Land holders. Farmer families in which one "
            "or more of its members belong to the following categories: former and present holders of "
            "constitutional posts, former and present Ministers, persons who paid income tax in the last "
            "assessment year."
        ),
    ),
    dict(
        title="Pradhan Mantri Fasal Bima Yojana | Vikaspedia - English - Agriculture",
        section="Objectives",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/agri-insurance/pradhan-mantri-fasal-bima-yojana",
        content=(
            "To provide insurance coverage and financial support to the farmers in the event of failure "
            "of any of the notified crop as a result of natural calamities, pests & diseases. To stabilise "
            "the income of farmers to ensure their continuance in farming. To encourage farmers to adopt "
            "innovative and modern agricultural practices."
        ),
    ),
    dict(
        title="Pradhan Mantri Fasal Bima Yojana | Vikaspedia - English - Agriculture",
        section="Highlights of the scheme",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/agri-insurance/pradhan-mantri-fasal-bima-yojana",
        content=(
            "There will be a uniform premium of only 2% to be paid by farmers for all Kharif crops and "
            "1.5% for all Rabi crops. In case of annual commercial and horticultural crops, the premium "
            "to be paid by farmers will be only 5%. The premium rates to be paid by farmers are very low "
            "and the balance premium will be paid by the Government to provide the full insured amount."
        ),
    ),
    dict(
        title="Pradhan Mantri Fasal Bima Yojana | Vikaspedia - English - Agriculture",
        section="Farmers to be covered",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/agri-insurance/pradhan-mantri-fasal-bima-yojana",
        content=(
            "All farmers growing notified crops in a notified area during the season who have insurable "
            "interest in the crop are eligible. The scheme has been made voluntary for all farmers from "
            "Kharif 2020."
        ),
    ),
    dict(
        title="Soil Health Card, Soil Conservation and Micronutrients | Vikaspedia - English - Agriculture",
        section="What Can You Get?",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/national-schemes-for-farmers/soil-health-card-soil-conservation-and-micronutrients",
        content=(
            "Assistance for Soil Improvement: Alkaline / Saline Soil - Rs. 60,000 per ha. Acidic Soil - "
            "Rs. 15,000 per ha. 44% of the cost, limited to Rs 44,000 per lab for individuals/private "
            "agencies through NABARD as capital investment for 3000 TPA production capacity."
        ),
    ),
    dict(
        title="Revised Kisan Credit Card Scheme | Vikaspedia - English - Agriculture",
        section="Objective",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/agri-credit/revised-kisan-credit-card-scheme",
        content=(
            "The scheme aims at providing adequate and timely credit for the comprehensive credit "
            "requirements of farmers under a single window for their cultivation and other needs: to "
            "meet the short term credit requirements for cultivation of crops, post harvest expenses, "
            "produce marketing loan, and consumption requirements of farmer households."
        ),
    ),
    dict(
        title="Revised Kisan Credit Card Scheme | Vikaspedia - English - Agriculture",
        section="Eligibility and credit limit",
        source_url="https://en.vikaspedia.in/viewcontent/agriculture/agri-credit/revised-kisan-credit-card-scheme",
        content=(
            "All farmers - individuals/joint borrowers who are owner cultivators. Tenant farmers, oral "
            "lessees and share croppers etc. SHGs or Joint Liability Groups of farmers including tenant "
            "farmers and share croppers are eligible."
        ),
    ),
    dict(
        title="PM KUSUM scheme | Vikaspedia - English - Energy",
        section="Scheme Components",
        source_url="https://en.vikaspedia.in/viewcontent/energy/policy-support/renewable-energy-1/solar-energy/pm-kusum-scheme",
        content=(
            "The Scheme consists of three components: Component A: Setting up of 10,000 MW of "
            "Decentralized Ground/Stilt Mounted Grid Connected Solar or other Renewable Energy based "
            "Power Plants by the farmers on their land. Component B: Installation of 14 Lakh Stand-alone "
            "Solar Agriculture Pumps. Component C: Solarisation of 35 Lakh Grid Connected Agriculture pumps."
        ),
    ),
]


def _load_primary() -> list[RawDocument] | None:
    if not PRIMARY_DATASET.exists():
        return None
    data = json.loads(PRIMARY_DATASET.read_text(encoding="utf-8"))
    return [RawDocument(**d) for d in data]


def _build_fallback() -> list[RawDocument]:
    from app.rag.metadata import new_id

    docs = []
    for item in _FALLBACK_DOCS:
        docs.append(
            RawDocument(
                document_id=new_id("demo"),
                title=item["title"],
                content=item["content"],
                source_type="vikaspedia",
                source_url=item["source_url"],
                section=item["section"],
                category="policies-and-schemes",
                language="en",
            )
        )
    return docs


def main() -> None:
    documents = _load_primary()
    if documents:
        print(f"Seeding demo knowledge base from real crawl output: {len(documents)} section-documents "
              f"({PRIMARY_DATASET}).")
    else:
        documents = _build_fallback()
        print(f"{PRIMARY_DATASET} not found — seeding demo knowledge base from {len(documents)} "
              f"embedded real Vikaspedia excerpts (offline-safe fallback).")

    stats = ingest_documents(documents)
    print(f"Demo knowledge base ready: {stats}")
    print("Run: streamlit run app/ui/streamlit_app.py")


if __name__ == "__main__":
    main()
