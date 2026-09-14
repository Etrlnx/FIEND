# Financial Intelligence and Evidence Nexus Decisioning (FIEND)

Created as an enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for high-stakes financial document intelligence and forensic regulatory analysis. Built on LangChain, FIEND ingests and processes regulatory corporate disclosures (SEC EDGAR Forms 10-K and 10-Q) to deliver mathematically grounded, citation-backed answers with strict refusal protocols and end-to-end evidence attribution.

## Features

- **SEC EDGAR Ingestion**: Downloads and parses 10-K/10-Q filings from SEC EDGAR
- **Table-Aware Processing**: Extracts and preserves financial tables as Markdown with context headers
- **Section-Aware Chunking**: Respects SEC filing structure (Part I/II, Item 1A, Item 7, etc.)
- **Hybrid Retrieval**: Combines dense (BGE-base) and sparse (BM25) retrieval with RRF
- **Metadata Filtering**: Pre-retrieval filtering by ticker, form, fiscal year, section
- **Cross-Encoder Reranking**: ms-marco-MiniLM-L-6-v2 reranker (top-20 → top-5)
- **Grounded Generation**: Hardened prompt with mandatory citations and refusal handling
- **Pluggable LLM Providers**: Gemini, Anthropic, OpenAI, Ollama (local, zero-quota)

---

## System Diagrams

### 1. Core RAG Model Overview

A high-level view of the Retrieval-Augmented Generation loop — showing the conceptual flow from a user question to a grounded, cited answer, without drilling into sub-components.

```mermaid
flowchart TD
    A([User Question]) --> B[Retriever]
    B --> C[(Knowledge Base\nSEC Filings)]
    C --> D[Relevant Passages]
    D --> E[Prompt Builder\nContext + Question]
    E --> F[LLM Generator]
    F --> G([Cited Answer])

    style A fill:#4A90D9,color:#fff,stroke:none
    style G fill:#27AE60,color:#fff,stroke:none
    style C fill:#F39C12,color:#fff,stroke:none
    style F fill:#8E44AD,color:#fff,stroke:none
```

---

### 2. Retrieval Pipeline

The multi-stage retrieval pipeline that narrows thousands of chunks down to the top-5 most relevant passages for any given query.

```mermaid
flowchart TD
    Q([User Query]) --> MF

    subgraph FILTER["Pre-Retrieval Metadata Filter"]
        MF["MetadataFilter\nextract_metadata_filter\nticker / form / year / section"]
    end

    MF --> DENSE
    MF --> BM25

    subgraph HYBRID["Hybrid Retrieval — Ensemble RRF"]
        DENSE["Dense Vector Search\nFAISS · BGE-base-en-v1.5\nweight = 0.7"]
        BM25["Sparse Keyword Search\nBM25Retriever\nweight = 0.3"]
        RRF["Reciprocal Rank Fusion\nRRF score = sum w / k + rank\ntop-20 candidates"]
        DENSE --> RRF
        BM25 --> RRF
    end

    RRF --> FR

    subgraph RERANK["Cross-Encoder Reranking"]
        FR["FilteredRetriever\nover-fetch 6x"]
        CE["CrossEncoder\nms-marco-MiniLM-L-6-v2\ntop-20 to top-5"]
        FR --> CE
    end

    CE --> R([Top-5 Ranked Passages])

    style Q fill:#4A90D9,color:#fff,stroke:none
    style R fill:#27AE60,color:#fff,stroke:none
```

---

### 3. FAISS Vector Store — Embedding & Indexing

How raw SEC filing HTML is transformed into searchable dense-vector embeddings and stored in the FAISS index.

```mermaid
flowchart LR
    subgraph INGEST["Document Ingestion"]
        A["SEC EDGAR\n10-K / 10-Q HTML"] --> B["loader.py\nHTML to LangChain Documents"]
        B --> C["table_extractor.py\nExtract tables as Markdown"]
        C --> D["chunking.py / splitter.py\nSection-aware recursive chunking"]
    end

    subgraph EMBED["Embedding"]
        D --> E["BGE-base-en-v1.5\nHuggingFace Embeddings\n768-dim vectors"]
    end

    subgraph STORE["FAISS Index"]
        E --> F["FAISS.from_documents\nbuild_vector_store"]
        F --> G[("FAISS Index\n16,626 text chunks\n4,678 table chunks\nsaved to disk")]
    end

    subgraph META["Metadata per Chunk"]
        H["ticker\nform\nfiling_date\nsection\nis_table"]
    end

    D --> META
    META --> G

    style A fill:#E74C3C,color:#fff,stroke:none
    style G fill:#F39C12,color:#fff,stroke:none
    style E fill:#8E44AD,color:#fff,stroke:none
```

---

### 4. Gemini & Ollama — LLM Generation with Grounded Prompting

How user queries and retrieved passages are assembled into a hardened prompt and dispatched to either the cloud (Gemini) or local (Ollama) LLM provider, with rate-limiting and retry logic.

```mermaid
flowchart TD
    subgraph RETRIEVAL["Retrieval Output"]
        P["Top-5 Ranked Passages\nwith metadata citations"]
    end

    subgraph PROMPT["Prompt Assembly — build_rag_prompt"]
        FD["format_docs\nNumbered passages with\nTICKER FORM DATE SECTION"]
        PT["RAG_PROMPT_HARDENED\nChatPromptTemplate\ncontext + question + citation rules"]
        FD --> PT
    end

    P --> FD
    Q([User Question]) --> PT

    subgraph PROVIDERS["Pluggable LLM Providers"]
        direction LR
        GEM["GeminiProvider\ngoogle-generativeai\ngemini-2.0-flash\nAPI key auth"]
        OLL["OllamaProvider\nlangchain-ollama\nllama3.2 local\nzero-quota"]
        ANT["AnthropicProvider\nclaude-3"]
        OAI["OpenAIProvider\ngpt-4o"]
    end

    subgraph RATELIMIT["Rate Limiting & Retry — create_rate_limited_llm"]
        RL["RunnableLambda\nRPM throttle: 60 / rpm seconds\ntenacity exponential backoff\n5 retries on 429 / quota errors"]
    end

    PT --> RATELIMIT
    RATELIMIT --> PROVIDERS

    PROVIDERS --> OUT

    subgraph OUT["Output Parsing"]
        SP["StrOutputParser to plain text"]
    end

    OUT --> A([Cited Answer\nAnswer with inline citations\nEvidence list])

    style Q fill:#4A90D9,color:#fff,stroke:none
    style GEM fill:#1A73E8,color:#fff,stroke:none
    style OLL fill:#2ECC71,color:#fff,stroke:none
    style A fill:#27AE60,color:#fff,stroke:none
```

---

### 5. Full-Stack Deployment Architecture

End-to-end containerised deployment with the FastAPI backend, Streamlit UI, PostgreSQL persistence, and the Prometheus → Loki → Grafana observability stack — all wired through a single Docker network.

```mermaid
flowchart TD
    USER(["User Browser"]) --> UI

    subgraph DOCKER["Docker Network — finrag-network"]

        subgraph APP["Application Layer"]
            UI["finrag-streamlit\nStreamlit UI\nport 8501"]
            API["finrag-api\nFastAPI Backend\nport 8000"]
            UI -- "REST / HTTP" --> API
        end

        subgraph DATA["Data Layer"]
            PG["finrag-postgres\nPostgres 16\nport 5432\nchat history and sessions"]
            FS[("FAISS Index\n/app/data/vector_stores\nBind Mount")]
        end

        subgraph OBS["Observability Stack"]
            PROM["finrag-prometheus\nMetrics scrape\nport 9090"]
            LOKI["finrag-loki\nLog aggregation\nport 3100"]
            PT2["finrag-promtail\nLog shipper\nDocker socket"]
            GRAF["finrag-grafana\nDashboards\nport 3000"]
            PT2 --> LOKI
            PROM --> GRAF
            LOKI --> GRAF
        end

        API --> PG
        API --> FS
        API --> PROM
        API --> LOKI
    end

    subgraph HOST["Host Machine"]
        OLL_HOST["Ollama\nllama3.2\nlocalhost 11434"]
    end

    API -- "host.docker.internal" --> OLL_HOST

    style USER fill:#4A90D9,color:#fff,stroke:none
    style OLL_HOST fill:#2ECC71,color:#fff,stroke:none
    style PG fill:#F39C12,color:#fff,stroke:none
    style GRAF fill:#E74C3C,color:#fff,stroke:none
    style API fill:#8E44AD,color:#fff,stroke:none
    style UI fill:#1A73E8,color:#fff,stroke:none
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Download SEC filings
python scripts/run_pipeline.py build

# 4. Query the system
python scripts/run_pipeline.py query "What was Apple's revenue in Q3 2026?"
```

## Configuration

Configure via `.env` file:

```bash
# LLM Provider (gemini, anthropic, openai, ollama)
LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=llama3.2

# Embedding model
EMBEDDING_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Vector store
VECTOR_STORE_DIR=data/vector_stores/phase6_table_aware

# Rate limiting
LLM_RPM=10
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096
```

## Commands

```bash
# Build vector index from SEC filings
python scripts/run_pipeline.py build

# Run test queries
python scripts/run_pipeline.py test

# Interactive query mode
python scripts/run_pipeline.py query

# Single question
python scripts/run_pipeline.py query "What was Apple's revenue in Q3 2026?"
```

## Evaluation

```bash
# Retrieval evaluation (45 questions)
python eval/evaluate_retrieval.py

# Generation evaluation (87 questions)
python eval/evaluate_generation.py --store data/vector_stores/phase6_table_aware --k 5 --fetch 20
```

## Project Structure

```
src/finrag/
├── config.py              # Configuration management
├── data/                  # Document loading & chunking
│   ├── loader.py          # SEC HTML parsing
│   ├── splitter.py        # Recursive chunking
│   ├── chunking.py        # Fixed/recursive/section-aware chunking
│   ├── table_extractor.py # HTML table extraction
│   └── xbrl.py            # XBRL cleanup
├── embeddings/            # Embedding providers
├── vectorstore/           # FAISS operations
├── retrieval/             # Retrieval pipeline
│   ├── __init__.py        # BM25, hybrid, reranking, filtering
│   └── metadata_filter.py # Query-time metadata filtering
├── generation/            # LLM generation
│   └── __init__.py        # Prompts, providers, rate limiting
├── pipeline.py            # FinRAGPipeline class
└── cli.py                 # CLI entry point

eval/
├── evaluate_generation.py # Generation evaluation
├── evaluate_retrieval.py  # Retrieval metrics
├── evaluate_embeddings.py # Embedding comparison
├── evaluate_hybrid.py     # Hybrid retrieval eval
├── evaluate_reranking.py  # Reranker evaluation
├── evaluate_filtering.py  # Metadata filtering eval
├── evaluate_tables.py     # Table extraction eval
├── evaluate_reranking.py  # Reranker evaluation
├── create_eval_set.py     # Eval set creation
└── add_table_questions.py # Table question generation
```

## Architecture

```
SEC EDGAR → HTML Loader → Table Extractor → Section-Aware Chunking
    → BGE-base Embeddings → FAISS Index (16,626 chunks, 4,678 tables)
    → Hybrid Retrieval (Dense 0.7 / BM25 0.3)
    → Metadata Filtering (ticker/form/year/section)
    → Cross-Encoder Reranker (top-20 → top-5)
    → Hardened Prompt + Ollama LLM
```

## Phase 9 Evaluation Results (87 questions)

| Config | Accuracy | Format % | Citation % | Grounded % | Refusal % |
|----------|----------|----------|------------|------------|-----------|
| Hardened + Rerank | 51.7% | 66.7% | 75.9% | 85.1% | 89.7% |
| Hardened + No Rerank | 54.0% | 63.2% | 74.7% | 94.3% | 82.8% |

See `eval/results/phase9_summary.md` for full results.

## Development

```bash
# Run tests
pytest tests/

# Run retrieval evaluation
python eval/evaluate_retrieval.py

# Run generation evaluation
python eval/evaluate_generation.py --store data/vector_stores/phase6_table_aware --k 5 --fetch 20
```

## Requirements

- Python 3.10+
- Ollama (for local LLM) or API keys for cloud providers
- ~4GB RAM for embeddings + FAISS index
- ~2GB disk for SEC filings + vector store

## License

MIT License - see LICENSE file for details.