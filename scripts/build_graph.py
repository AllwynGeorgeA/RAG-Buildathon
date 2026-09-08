#!/usr/bin/env python
"""
Rebuilds the knowledge graph from ingested documents WITHOUT touching the
vector store. Useful after editing the entity-extraction rules in
app/graph/graph_builder.py, to refresh the graph without re-embedding.

Usage:
    python scripts/build_graph.py --input data/raw/vikaspedia_documents.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.graph_builder import build_graph_from_documents  # noqa: E402
from app.graph.knowledge_graph import KnowledgeGraph  # noqa: E402
from app.rag.metadata import RawDocument  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the NetworkX knowledge graph.")
    parser.add_argument("--input", type=str, default="data/raw/vikaspedia_documents.json")
    parser.add_argument("--fresh", action="store_true", help="Start from an empty graph instead of merging.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}.")
        raise SystemExit(1)

    documents = [RawDocument(**d) for d in json.loads(input_path.read_text(encoding="utf-8"))]
    kg = KnowledgeGraph() if args.fresh else KnowledgeGraph.load()
    kg = build_graph_from_documents(documents, kg=kg)
    path = kg.save()
    print(f"Graph rebuilt: {kg.node_count()} nodes, {kg.edge_count()} edges -> {path}")


if __name__ == "__main__":
    main()
