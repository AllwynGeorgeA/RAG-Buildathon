"""
Offline Vikaspedia crawler (Playwright).

Runs ONLY via `scripts/crawl_vikaspedia.py` — never during a chat request
(spec section 6/15). Vikaspedia's pages are client-rendered, so a plain HTTP
GET returns an empty shell; Playwright renders the page before we extract
content.

Extraction strategy: walk the rendered page's heading structure (h2/h3) and
emit one RawDocument PER SECTION (not per page), so `doc.section` carries
the actual heading text ("What are the benefits?", "Eligibility", ...). This
is what lets `app/graph/graph_builder.py` attach benefits/eligibility to the
right scheme deterministically.
"""
from __future__ import annotations

import json
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

from app.core.exceptions import CrawlPolicyError
from app.core.logging import get_logger
from app.ingestion.crawler_config import CrawlConfig, is_url_allowed, load_crawl_config
from app.rag.metadata import RawDocument, content_hash, new_id

logger = get_logger(__name__)

_LANG_SUBDOMAIN_MAP = {
    "en": "en", "hi": "hi", "ta": "ta", "te": "te", "kn": "kn", "ml": "ml",
    "mr": "mr", "bn": "bn", "gu": "gu", "pa": "pa", "or": "or", "as": "as",
}


@dataclass
class CrawlManifestEntry:
    url: str
    content_hash: str
    crawl_timestamp: str
    document_ids: list[str] = field(default_factory=list)


class CrawlManifest:
    """Tracks visited URLs + content hashes for incremental, idempotent crawling."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, CrawlManifestEntry] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for url, entry in data.items():
                self.entries[url] = CrawlManifestEntry(**entry)

    def has_unchanged(self, url: str, new_hash: str) -> bool:
        existing = self.entries.get(url)
        return bool(existing and existing.content_hash == new_hash)

    def record(self, entry: CrawlManifestEntry) -> None:
        self.entries[entry.url] = entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {url: vars(e) for url, e in self.entries.items()}
        self.path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


class RobotsCache:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin(origin, "/robots.txt"))
            try:
                rp.read()
            except Exception:  # noqa: BLE001 — if robots.txt is unreachable, default to allow
                logger.warning("Could not fetch robots.txt — defaulting to allow", extra={"origin": origin})
                self._parsers[origin] = None  # type: ignore[assignment]
                return True
            self._parsers[origin] = rp
        rp = self._parsers[origin]
        return rp.can_fetch(self.user_agent, url) if rp else True


_BOILERPLATE_HEADING_MARKERS = {"government of india", "table of contents", "about us", "contact us"}
_BOILERPLATE_TEXT_MARKERS = [
    "skip the lengthy reading", "add vikaspedia as a trusted source", "contributor",
    "translate to", "summarize content", "vikas ai",
    "report page", "ratings (", "reviews", "view reply", "1 star", "2 stars",
]
# A page's user-comment thread ("Dr Nagsen Meshram ... 6/15/2022 ... View
# Reply") reads as real prose to a naive extractor but is NOT scheme content —
# match a date-like token near a short exchange as a strong comment-thread signal.
_COMMENT_THREAD_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}")


def _is_boilerplate_section(heading: str, text: str) -> bool:
    """Drops nav/footer/site-furniture/comment-thread sections picked up
    alongside real content so they don't pollute the index."""
    heading_l = heading.strip().lower()
    text_l = text.strip().lower()
    if heading_l in _BOILERPLATE_HEADING_MARKERS:
        return True
    if any(marker in text_l[:160] for marker in _BOILERPLATE_TEXT_MARKERS):
        return True
    if _COMMENT_THREAD_RE.search(text):
        return True
    if len(text.strip()) < 20 and not heading:
        return True
    return False


def _detect_language(url: str) -> str:
    host = urlparse(url).netloc.split(".")[0]
    return _LANG_SUBDOMAIN_MAP.get(host, "en")


def _detect_category(url: str) -> str:
    path = urlparse(url).path.lower()
    if "scheme" in path:
        return "policies-and-schemes"
    if "agriculture" in path:
        return "agriculture"
    return "general"


def _extract_sections(page) -> list[dict]:
    """
    Runs in-browser JS to split the rendered article body into
    {heading, text, anchor} sections based on h2/h3 headings.
    Falls back to a single section (heading="") if no headings are found.
    """
    return page.evaluate(
        """
        () => {
          const root = document.querySelector('main, article, #content, .content, body');
          if (!root) return [];
          const nodes = Array.from(root.querySelectorAll('h1, h2, h3, p, li'));
          const sections = [];
          let current = { heading: '', text: [], anchor: '' };
          for (const node of nodes) {
            const tag = node.tagName.toLowerCase();
            const text = node.innerText ? node.innerText.trim() : '';
            if (!text) continue;
            if (tag === 'h1' || tag === 'h2' || tag === 'h3') {
              if (current.text.length) sections.push(current);
              current = { heading: text, text: [], anchor: node.id || '' };
            } else {
              current.text.push(text);
            }
          }
          if (current.text.length) sections.push(current);
          return sections.map(s => ({ heading: s.heading, text: s.text.join('\\n'), anchor: s.anchor }));
        }
        """
    )


def _extract_links(page, base_url: str) -> list[str]:
    hrefs = page.evaluate(
        "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href'))"
    )
    links = []
    for href in hrefs or []:
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        links.append(urljoin(base_url, href))
    return links


def crawl(
    config: CrawlConfig | None = None,
    manifest_path: Path | None = None,
    max_pages: int | None = None,
) -> list[RawDocument]:
    """
    Crawl seed URLs breadth-first up to max_depth, extracting section-level
    RawDocuments. Returns the list of NEW/CHANGED documents (unchanged pages
    are skipped via the manifest for incremental crawling).
    """
    from playwright.sync_api import sync_playwright

    config = config or load_crawl_config()
    max_pages = max_pages or config.max_pages
    manifest_path = manifest_path or Path("data/raw/crawl_manifest.json")
    manifest = CrawlManifest(manifest_path)
    robots = RobotsCache(config.user_agent)

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(url, 0) for url in config.seed_urls]
    documents: list[RawDocument] = []
    pages_fetched = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=config.user_agent)
        page = context.new_page()

        while queue and pages_fetched < max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > config.max_depth:
                continue
            visited.add(url)

            if not is_url_allowed(url, config):
                continue
            if config.respect_robots_txt and not robots.allowed(url):
                logger.info("Blocked by robots.txt", extra={"url": url})
                continue

            success = False
            last_error = None
            for attempt in range(config.max_retries):
                try:
                    page.goto(url, timeout=config.request_timeout_seconds * 1000, wait_until="networkidle")
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(1.5 * (attempt + 1))
            if not success:
                logger.warning("Failed to fetch page after retries", extra={"url": url, "error": str(last_error)})
                continue

            pages_fetched += 1
            title = page.title() or url
            sections = _extract_sections(page)
            full_text_hash = content_hash(title + "".join(s["text"] for s in sections))

            if manifest.has_unchanged(url, full_text_hash):
                logger.info("Unchanged since last crawl — skipping", extra={"url": url})
            else:
                page_document_ids = []
                for section in sections:
                    if not section["text"].strip():
                        continue
                    if _is_boilerplate_section(section["heading"], section["text"]):
                        continue
                    doc = RawDocument(
                        document_id=new_id("vikaspedia"),
                        title=title,
                        content=section["text"],
                        source_type="vikaspedia",
                        source_url=(url + (f"#{section['anchor']}" if section["anchor"] else "")),
                        language=_detect_language(url),
                        sector="agriculture",
                        category=_detect_category(url),
                        section=section["heading"],
                        file_type="html",
                    )
                    documents.append(doc)
                    page_document_ids.append(doc.document_id)

                manifest.record(
                    CrawlManifestEntry(
                        url=url,
                        content_hash=full_text_hash,
                        crawl_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        document_ids=page_document_ids,
                    )
                )

            if depth < config.max_depth:
                for link in _extract_links(page, url):
                    if link not in visited and is_url_allowed(link, config):
                        queue.append((link, depth + 1))

            time.sleep(config.rate_limit_seconds)

        browser.close()

    manifest.save()
    logger.info("Crawl complete", extra={"pages_fetched": pages_fetched, "new_documents": len(documents)})
    return documents
