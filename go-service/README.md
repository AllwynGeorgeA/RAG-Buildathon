# KrishiMitra AI — Go ingestion service (optional)

A small, dependency-free (stdlib only) Go service for **crawl/ingestion job
orchestration**. It is optional — the Python API and Streamlit UI work fully
without it. It exists only for the high-concurrency edge of the system:
queuing multiple crawl/ingest jobs and running them in parallel via worker
goroutines, without blocking the Python process.

It does **not** duplicate any RAG/LLM/embedding/graph/eligibility logic —
every job it runs simply shells out to the real Python scripts
(`scripts/crawl_vikaspedia.py`, `scripts/build_index.py`), so there is one
implementation of the ingestion pipeline, not two.

## Build & run

```bash
cd go-service
go build -o bin/ingestion-service .
KRISHIMITRA_REPO_DIR=.. KRISHIMITRA_PYTHON_BIN=../.venv/Scripts/python ./bin/ingestion-service
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8090` | HTTP listen port |
| `KRISHIMITRA_REPO_DIR` | `..` | Working directory the Python scripts run from |
| `KRISHIMITRA_PYTHON_BIN` | `python` | Python interpreter to invoke (point at the venv) |

## Endpoints

- `GET /health` — liveness check
- `POST /jobs` — enqueue a job: `{"type": "crawl", "payload": {"max_pages": "50"}}` or `{"type": "ingest", "payload": {"documents_path": "data/raw/vikaspedia_documents.json"}}`
- `GET /jobs` — list all jobs and their status
- `GET /jobs/{id}` — poll one job's status/output

## Note on this build

This service was authored to compile with the Go standard library only
(`net/http`, `encoding/json`, `os/exec`, `sync`) — no external modules, so
`go build` needs no network access. It has **not** been compiled/run in this
development environment (no Go toolchain was available), so treat it as
reference code to verify with `go vet`/`go build` before relying on it.
