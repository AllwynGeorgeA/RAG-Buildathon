#!/usr/bin/env python
"""
Builds/updates the vector store from crawled or JSON-serialized RawDocuments.

Usage:
    python scripts/build_index.py --input data/raw/vikaspedia_documents.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.pipeline import ingest_documents  # noqa: E402
from app.rag.metadata import RawDocument  # noqa: E402


def load_raw_documents(path: Path) -> list[RawDocument]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RawDocument(**d) for d in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local vector index from ingested documents.")
    parser.add_argument("--input", type=str, default="data/raw/vikaspedia_documents.json")
    parser.add_argument("--no-graph", action="store_true", help="Skip knowledge graph update (index only).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}. Run scripts/crawl_vikaspedia.py first, "
              f"or scripts/demo_seed.py for demo data.")
        raise SystemExit(1)

    documents = load_raw_documents(input_path)
    print(f"Loaded {len(documents)} documents. Embedding + indexing...")
    stats = ingest_documents(documents, update_graph=not args.no_graph)
    print(f"Done. {stats}")


if __name__ == "__main__":
    main()
