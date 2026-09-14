# FinRAG: Budget, Rate Limits, and Safety Ceilings

This document outlines API limits, request throttling, and cost estimation for the FinRAG pipeline across all experimental phases.

---

## 1. SEC EDGAR API Rate Limits & Compliance

- **SEC Rule**: Maximum **10 requests per second** per IP address.
- **Fair Access Requirement**: A descriptive `User-Agent` header must accompany every request:
  ```http
  User-Agent: SampleCompanyName AdminContact@sampledomain.com
  ```
- **FinRAG Implementation**:
  - `src/data/edgar_downloader.py` enforces an automatic rate-limiting delay (`0.15s` - `0.2s` between requests) to stay comfortably below 10 req/sec.
  - SEC EDGAR data is free and legally public; all raw documents are stored in `data/raw/` so no repeated network calls are made during RAG experiments.

---

## 2. Embedding API Budget & Cost Calculations

| Provider / Model | Dimensions | Price per 1M Tokens (Approx) | Estimated Cost for 30 Filings (~1.5M tokens) |
| :--- | :--- | :--- | :--- |
| **OpenAI text-embedding-3-small** | 1536 | $0.02 / 1M tokens | ~$0.03 |
| **OpenAI text-embedding-3-large** | 3072 | $0.13 / 1M tokens | ~$0.20 |
| **Google text-embedding-004** | 768 | Free Tier / $0.000025 / 1k chars | ~$0.04 |
| **HuggingFace / BGE-small / MiniLM** | 384-768 | Local / Free (CPU/GPU) | $0.00 |

> **Embedding Cache Rule**: We will persist chunk embeddings in local FAISS/Chroma vector store indexes in `data/vector_stores/` so vector computation is performed **once** per experiment configuration, never re-embedded needlessly.

---

## 3. LLM API Rate Limits & Evaluation Budget

| Phase | Description | Estimated LLM Calls | Estimated Token Volume | Estimated Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1 (Baseline)** | 10 sanity queries | 10 | ~25k tokens | < $0.05 |
| **Phase 2–7 (Retrieval Exp)** | Retrieval metrics only | 0 LLM calls (pure retrieval) | 0 | $0.00 |
| **Phase 8 (Grounded Prompting)** | 20 prompt runs | 20 | ~50k tokens | < $0.10 |
| **Phase 9 (Evaluation Benchmark)**| 30–50 Q&A eval runs + RAGAS judging | ~100–150 calls | ~400k tokens | ~$0.50 – $1.50 |

---

## 4. Rate-Limiting & Safety Recommendations

1. **Local Vector Caching**: Store all generated vector indexes under `data/vector_stores/` keyed by chunking method + embedding model name.
2. **Evaluation Batching**: When running evaluation harnesses (RAGAS or LLM judge), throttle parallel requests with a concurrency limit of 5 workers to avoid hitting provider TPM (tokens per minute) rate limits.
3. **Hard Ceiling**: Total API expenditure for the entire research phases (0–11) is estimated to be well under **$5.00 USD** if using OpenAI `gpt-4o-mini` / Google `gemini-1.5-flash` for generation & evaluation.
