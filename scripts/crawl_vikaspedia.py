#!/usr/bin/env python
"""
Offline Vikaspedia crawl entry point. Run explicitly — NEVER invoked during
a chat session (spec section 6).

Usage:
    python scripts/crawl_vikaspedia.py [--max-pages N] [--config path/to/crawl.yaml]

Requires `playwright install chromium` to have been run once beforehand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import get_logger  # noqa: E402
from app.ingestion.crawler_config import load_crawl_config  # noqa: E402
from app.ingestion.vikaspedia_crawler import crawl  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Vikaspedia for farmer scheme content.")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/raw/vikaspedia_documents.json")
    args = parser.parse_args()

    config = load_crawl_config(args.config)
    print(f"Starting crawl — seeds: {len(config.seed_urls)}, max_depth: {config.max_depth}, "
          f"max_pages: {args.max_pages or config.max_pages}")

    documents = crawl(config=config, max_pages=args.max_pages)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([vars(d) for d in documents], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Crawl finished. {len(documents)} section-documents saved to {out_path}")
    print("Next: python scripts/build_index.py  &&  python scripts/build_graph.py")


if __name__ == "__main__":
    main()
