# Vector Database Research & Recommendation

## Overview

This document compares four candidate vector databases for the Agentic RAG pipeline
used in the Bahasa Indonesia disinformation, defamation, and hate speech detection system.
The evaluation prioritizes **local development ease**, **metadata filtering**, and
**Python SDK quality**.

---

## Candidates Evaluated

| Criteria | ChromaDB | Qdrant | Weaviate (local) | FAISS |
|---|---|---|---|---|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ Pip install, zero config | ⭐⭐⭐⭐ Docker needed | ⭐⭐⭐ Docker + config file | ⭐⭐⭐ CPU/GPU setup manual |
| **Local Support** | ✅ Fully embedded, persisted to disk | ✅ Docker or embedded mode | ✅ Docker only | ✅ Fully in-process |
| **Metadata Filtering** | ✅ `where` dict syntax, rich filters | ✅ Advanced payload filtering | ✅ GraphQL-based, very expressive | ❌ No native metadata filtering |
| **Python SDK Quality** | ✅ Pythonic, well-documented | ✅ Comprehensive, typed | ✅ Good but verbose | ⚠️ Low-level C++ wrapper |
| **Persistence** | ✅ SQLite/DuckDB on disk | ✅ Wal-based snapshots | ✅ Volume-based | ❌ No built-in persistence |
| **Scalability** | ⚠️ Single-node only | ✅ Distributed capable | ✅ Cluster mode | ⚠️ Memory-bound |
| **Licensing** | Apache 2.0 (open source) | Apache 2.0 | BSD | MIT |
| **Active Maintenance** | ✅ Very active | ✅ Very active | ✅ Active | ✅ Meta-maintained |

---

## Detailed Analysis

### 1. ChromaDB

**Setup**:
```bash
pip install chromadb
```
No additional infrastructure required. Works both in-memory and with persistent storage via
SQLite/DuckDB.

**Pros**:
- The simplest possible setup for local development
- Rich Python-native API: `collection.query(query_embeddings=..., where={...})`
- Supports `where` metadata filters (e.g., filter by document source, category, date)
- Built-in embedding functions (can plug in any embedding model)
- Good documentation and growing community

**Cons**:
- Not suited for production-scale distributed deployments without a separate server
- Search performance degrades at very large scale (millions of vectors)

**Metadata Filtering Example**:
```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={"source": "kominfo", "category": "hoaks"}
)
```

---

### 2. Qdrant

**Setup**:
```bash
docker run -p 6333:6333 qdrant/qdrant
```
Or use the in-process Python client (`qdrant_client.QdrantClient(":memory:")` or local path).

**Pros**:
- Excellent filtering via "payload" system
- Supports hybrid search (sparse + dense)
- Very high performance at scale

**Cons**:
- Docker preferred for production; in-memory mode has limitations
- Slightly more complex API than ChromaDB

---

### 3. Weaviate (local)

**Setup**:
```bash
docker compose up -d  # requires YAML config
```

**Pros**:
- Very expressive schema + filtering via GraphQL
- Multi-modal support

**Cons**:
- Significantly more complex setup
- GraphQL overhead for simple use cases
- Heavier resource footprint

---

### 4. FAISS

**Setup**:
```bash
pip install faiss-cpu
```

**Pros**:
- Extremely fast for ANN (Approximate Nearest Neighbor) search
- Industry-standard, used by Meta

**Cons**:
- **No built-in metadata filtering** — requires a parallel data store (e.g., SQLite)
- No persistence out-of-the-box (must serialize index manually)
- Low-level C++ API wrapped in Python — less ergonomic

---

## ✅ Final Recommendation: **ChromaDB**

**Chosen for this project** based on the following rationale:

| Factor | Justification |
|---|---|
| **Ease of Setup** | Single `pip install`, no Docker, no daemon. Works immediately in any Python environment. |
| **Local Support** | Persists to disk automatically via SQLite — no data loss between runs. |
| **Metadata Filtering** | Native `where` clause supports filtering by source, category, language, date — essential for separating hoax from hate speech documents. |
| **Python SDK Quality** | Clean, Pythonic API that integrates naturally with custom embedding models and supports `add`, `query`, `delete`, and `update` operations with minimal boilerplate. |
| **Development Speed** | Minimal infrastructure complexity lets the team focus on pipeline logic rather than infrastructure management. |

ChromaDB is the optimal choice for a **local/dev-first** RAG pipeline. If the project
scales to production serving thousands of users, migrating to Qdrant (with its Docker
deployment and superior distributed filtering) is the recommended upgrade path.

---

## Implementation Notes

```python
import chromadb

# Initialize persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or load collection
collection = client.get_or_create_collection(
    name="indonesian_docs",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Insert document chunks
collection.add(
    documents=["chunk text..."],
    embeddings=[[0.1, 0.2, ...]],
    metadatas=[{"source": "kominfo", "category": "hoaks"}],
    ids=["doc1_chunk0"]
)

# Query with top-K retrieval
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)
```
