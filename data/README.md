# `data/` directory

| Path | Versioned? | Contents |
|---|---|---|
| `raw/vikaspedia_documents.json` | **Yes** | Output of a real `scripts/crawl_vikaspedia.py` run against live Vikaspedia — section-level documents with full provenance (`source_url`, `section`, `crawl_timestamp`, content hash). This is what makes `DEMO_MODE` work fully offline. |
| `raw/crawl_manifest.json` | No | Incremental-crawl bookkeeping (visited URLs + content hashes). Regenerated per crawl run. |
| `processed/*.json` | No | Per-document audit trail written by `app/ingestion/pipeline.py` during ingestion — one JSON file per ingested `RawDocument`. Regenerable from `raw/`. |
| `uploads/` | No | Scratch space for user uploads processed via the API/UI (files are deleted after processing; nothing should persist here). |
| `vector_store/` | No | Chroma's persistent local database (embeddings + chunk text + metadata). Rebuild with `python scripts/build_index.py` or `python scripts/demo_seed.py`. |
| `knowledge_graph/graph.gpickle` | No | The pickled NetworkX knowledge graph. Rebuild with `python scripts/build_graph.py` or `python scripts/demo_seed.py`. |

**Quick start from a fresh clone:**

```bash
python scripts/demo_seed.py   # ingests raw/vikaspedia_documents.json into a fresh vector_store + graph
streamlit run app/ui/streamlit_app.py
```
