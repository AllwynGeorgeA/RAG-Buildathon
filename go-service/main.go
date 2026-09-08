// KrishiMitra AI — optional Go ingestion-orchestration service.
//
// Python remains responsible for RAG, embeddings, the knowledge graph, the
// LLM, eligibility reasoning, and the Streamlit UI (see ../app/). This
// service exists only for the high-performance, concurrency-heavy edge of
// the system that Go suits well and Python doesn't need to own:
//
//   - a durable-ish, in-memory job queue for crawl/ingest jobs
//   - concurrent worker goroutines so multiple ingest jobs run in parallel
//     without blocking the API/UI process
//   - a lightweight health endpoint for ops/monitoring
//
// It never talks to the LLM, vector store, or knowledge graph directly —
// jobs are executed by shelling out to the same Python scripts a human
// operator would run (scripts/crawl_vikaspedia.py, scripts/build_index.py),
// so there is exactly one implementation of the ingestion pipeline, not two.
//
// Build:  cd go-service && go build -o bin/ingestion-service .
// Run:    ./bin/ingestion-service        (listens on :8090 by default)
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

type JobStatus string

const (
	StatusQueued  JobStatus = "queued"
	StatusRunning JobStatus = "running"
	StatusDone    JobStatus = "done"
	StatusFailed  JobStatus = "failed"
)

type JobType string

const (
	JobCrawl  JobType = "crawl"
	JobIngest JobType = "ingest"
)

// Job represents one crawl or ingest request. Payload fields are passed
// through as CLI flags to the corresponding Python script.
type Job struct {
	ID        string    `json:"id"`
	Type      JobType   `json:"type"`
	Payload   map[string]string `json:"payload,omitempty"`
	Status    JobStatus `json:"status"`
	Output    string    `json:"output,omitempty"`
	Error     string    `json:"error,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	StartedAt *time.Time `json:"started_at,omitempty"`
	EndedAt   *time.Time `json:"ended_at,omitempty"`
}

type JobQueue struct {
	mu      sync.Mutex
	jobs    map[string]*Job
	queue   chan *Job
	nextID  int
	repoDir string
	pythonBin string
}

func NewJobQueue(workers int, repoDir, pythonBin string) *JobQueue {
	q := &JobQueue{
		jobs:      make(map[string]*Job),
		queue:     make(chan *Job, 100),
		repoDir:   repoDir,
		pythonBin: pythonBin,
	}
	for i := 0; i < workers; i++ {
		go q.worker(i)
	}
	return q
}

func (q *JobQueue) Enqueue(jobType JobType, payload map[string]string) *Job {
	q.mu.Lock()
	q.nextID++
	job := &Job{
		ID:        fmt.Sprintf("job_%d", q.nextID),
		Type:      jobType,
		Payload:   payload,
		Status:    StatusQueued,
		CreatedAt: time.Now(),
	}
	q.jobs[job.ID] = job
	q.mu.Unlock()

	q.queue <- job
	return job
}

func (q *JobQueue) Get(id string) (*Job, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	job, ok := q.jobs[id]
	return job, ok
}

func (q *JobQueue) List() []*Job {
	q.mu.Lock()
	defer q.mu.Unlock()
	jobs := make([]*Job, 0, len(q.jobs))
	for _, j := range q.jobs {
		jobs = append(jobs, j)
	}
	return jobs
}

func (q *JobQueue) worker(id int) {
	for job := range q.queue {
		q.runJob(job)
	}
}

func (q *JobQueue) runJob(job *Job) {
	now := time.Now()
	q.mu.Lock()
	job.Status = StatusRunning
	job.StartedAt = &now
	q.mu.Unlock()

	var scriptArgs []string
	switch job.Type {
	case JobCrawl:
		scriptArgs = []string{"scripts/crawl_vikaspedia.py"}
		if maxPages, ok := job.Payload["max_pages"]; ok {
			scriptArgs = append(scriptArgs, "--max-pages", maxPages)
		}
	case JobIngest:
		scriptArgs = []string{"scripts/build_index.py"}
		if input, ok := job.Payload["documents_path"]; ok {
			scriptArgs = append(scriptArgs, "--input", input)
		}
	default:
		q.finish(job, "", fmt.Errorf("unknown job type: %s", job.Type))
		return
	}

	cmd := exec.Command(q.pythonBin, scriptArgs...)
	cmd.Dir = q.repoDir
	output, err := cmd.CombinedOutput()
	q.finish(job, string(output), err)
}

func (q *JobQueue) finish(job *Job, output string, err error) {
	now := time.Now()
	q.mu.Lock()
	defer q.mu.Unlock()
	job.Output = output
	job.EndedAt = &now
	if err != nil {
		job.Status = StatusFailed
		job.Error = err.Error()
	} else {
		job.Status = StatusDone
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func main() {
	repoDir := os.Getenv("KRISHIMITRA_REPO_DIR")
	if repoDir == "" {
		repoDir = ".."
	}
	pythonBin := os.Getenv("KRISHIMITRA_PYTHON_BIN")
	if pythonBin == "" {
		pythonBin = "python"
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = "8090"
	}

	queue := NewJobQueue(4, repoDir, pythonBin)

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"status":  "ok",
			"service": "krishimitra-ingestion-service",
			"time":    time.Now().UTC().Format(time.RFC3339),
		})
	})

	mux.HandleFunc("/jobs", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			var req struct {
				Type    JobType           `json:"type"`
				Payload map[string]string `json:"payload"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
				return
			}
			if req.Type != JobCrawl && req.Type != JobIngest {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": "type must be 'crawl' or 'ingest'"})
				return
			}
			job := queue.Enqueue(req.Type, req.Payload)
			writeJSON(w, http.StatusAccepted, job)
		case http.MethodGet:
			writeJSON(w, http.StatusOK, queue.List())
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/jobs/", func(w http.ResponseWriter, r *http.Request) {
		id := r.URL.Path[len("/jobs/"):]
		job, ok := queue.Get(id)
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "job not found"})
			return
		}
		writeJSON(w, http.StatusOK, job)
	})

	addr := ":" + port
	log.Printf("KrishiMitra ingestion service listening on %s (repo_dir=%s, python=%s)", addr, repoDir, pythonBin)
	log.Fatal(http.ListenAndServe(addr, mux))
}
