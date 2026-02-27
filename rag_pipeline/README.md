# 🛡️ Agentic RAG Pipeline — Deteksi Disinformasi Bahasa Indonesia

Sistem RAG (Retrieval-Augmented Generation) untuk mendeteksi **disinformasi**, **defamasi**, dan **ujaran kebencian** dalam Bahasa Indonesia.

---

## 📁 Struktur Proyek

```
rag_pipeline/
├── main.py                    # Entry point — jalankan pipeline lengkap
├── vector_store.py            # ChromaDB: setup, insert, cosine search
├── embedder.py                # Embedding model wrapper (3 model, mudah diganti)
├── retriever.py               # Query → embed → search → retrieve top-K
├── prompt_builder.py          # Inject retrieved chunks ke prompt template
├── generator.py               # Call Claude API → structured AnalysisResult
├── benchmark_embeddings.py    # Benchmark 3 embedding model (kecepatan + similarity)
├── docs/
│   ├── vector_db_research.md  # Komparasi & rekomendasi vector DB
│   └── embedding_research.md  # Komparasi & rekomendasi embedding model
├── requirements.txt
├── .env.example               # Template environment variables
└── README.md
```

---

## ⚡ Quickstart

### 1. Install Dependencies

```bash
cd rag_pipeline
pip install -r requirements.txt
```

> **Catatan**: `sentence-transformers` akan mengunduh model LaBSE (~470MB) atau
> multilingual-e5-large (~560MB) secara otomatis saat pertama kali dijalankan.

### 2. Setup Environment Variables

```bash
# Salin template .env
cp .env.example .env
```

Edit `.env` dan isi API keys Anda:

```env
ANTHROPIC_API_KEY=sk-ant-...       # Wajib untuk generasi jawaban
OPENAI_API_KEY=sk-...              # Opsional, untuk embedding model OpenAI
EMBEDDING_MODEL=multilingual-e5    # Pilihan: openai, labse, multilingual-e5
```

### 3. Jalankan Pipeline

```bash
python main.py
```

Output akan menampilkan:
- Klasifikasi: `DISINFORMATION / DEFAMATION / HATE_SPEECH / CLEAN / MULTIPLE`
- Confidence level dan score
- Reasoning berbasis dokumen referensi
- Sumber/referensi yang digunakan

### 4. Analisis Query Kustom

```bash
python main.py --query "teks atau klaim yang ingin dianalisis"

# Ganti embedding model via CLI:
python main.py --query "..." --model labse

# Force reload dokumen ke vector store:
python main.py --reload
```

---

## 🔄 Alur Pipeline

```
Input Query
    │
    ▼
[Embedder] Embed query → vector
    │
    ▼
[VectorStore] Cosine similarity search → Top-5 chunks
    │
    ▼
[PromptBuilder] Format chunks + inject ke prompt template
    │
    ▼
[Generator] Claude API → Structured JSON analysis
    │
    ▼
Output: Classification + Confidence + Reasoning + Sources
```

---

## 🔧 Konfigurasi

Semua konfigurasi diatur melalui file `.env`:

| Variable | Default | Keterangan |
|---|---|---|
| `EMBEDDING_MODEL` | `multilingual-e5` | Model embedding: `openai`, `labse`, `multilingual-e5` |
| `ANTHROPIC_API_KEY` | — | API key Anthropic (wajib) |
| `OPENAI_API_KEY` | — | API key OpenAI (wajib jika `EMBEDDING_MODEL=openai`) |
| `CLAUDE_MODEL` | `claude-3-5-sonnet-20241022` | Model Claude untuk generasi |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Direktori penyimpanan ChromaDB |
| `TOP_K` | `5` | Jumlah chunk yang diambil saat retrieval |

---

## 📊 Benchmark Embedding Models

Jalankan benchmark perbandingan 3 model embedding:

```bash
python benchmark_embeddings.py
```

Benchmark menguji:
- **Kecepatan**: Embed 100 kalimat Bahasa Indonesia
- **Semantic Similarity**: 5 pasang kalimat (similar vs dissimilar)

Lihat hasil lengkap di [`docs/embedding_research.md`](docs/embedding_research.md).

---

## 🗄️ Vector Database

Proyek ini menggunakan **ChromaDB** dengan alasan:

| Kriteria | ChromaDB |
|---|---|
| Setup | `pip install chromadb` — tanpa Docker |
| Storage | Persistent ke disk (SQLite/DuckDB) |
| Metadata filtering | ✅ `where` clause native |
| Python SDK | ✅ Pythonic dan well-documented |

Lihat komparasi lengkap di [`docs/vector_db_research.md`](docs/vector_db_research.md).

---

## 🧩 Mengganti Embedding Model

Cukup ubah `EMBEDDING_MODEL` di file `.env`:

```env
# Pilihan 1: OpenAI API (paling akurat, berbayar)
EMBEDDING_MODEL=openai

# Pilihan 2: LaBSE (cross-lingual, gratis, lokal)
EMBEDDING_MODEL=labse

# Pilihan 3: multilingual-e5-large (terbaik untuk retrieval, gratis, lokal) — RECOMMENDED
EMBEDDING_MODEL=multilingual-e5
```

> **Penting**: Jika mengganti model setelah data diindeks, jalankan dengan `--reload`
> untuk re-embed semua dokumen dengan model baru:
> ```bash
> python main.py --reload
> ```

---

## 📝 Sample Output

```
============================================================
📊 HASIL ANALISIS KONTEN
============================================================
🏷️  Klasifikasi  : DISINFORMATION
📈 Kepercayaan  : HIGH (92%)

📝 Penjelasan:
   Klaim bahwa vaksin mengandung microchip atau chip 5G merupakan
   disinformasi yang telah secara resmi dibantah oleh Kominfo.
   Berdasarkan dokumen referensi, ini adalah hoaks yang tidak
   memiliki dasar ilmiah.

🔍 Bukti Pendukung:
   [1] "Berdasarkan fact-check Kominfo, klaim bahwa vaksin mengandung
       chip 5G adalah hoaks yang telah dibantah secara ilmiah."
   [2] "MAFINDO mendefinisikan disinformasi sebagai informasi yang
       salah dan disebarkan dengan niat untuk menipu."

📚 Referensi Digunakan: doc1, doc5
💡 Rekomendasi: Jangan sebarkan konten ini. Rujuk ke sumber resmi
   seperti website Kominfo atau MAFINDO untuk klarifikasi.
============================================================
```

---

## 📚 Menambah Dokumen Referensi

```python
from vector_store import VectorStore

vs = VectorStore()
vs.add_document(
    doc_id="kominfo_2024_001",
    text="Isi dokumen fact-check atau regulasi...",
    metadata={
        "source": "kominfo",
        "category": "hoaks",       # hoaks | hate_speech | defamasi | disinformasi
        "topic": "vaksin",
        "language": "id",
    }
)
```

---

## 🛠️ Requirements

- Python 3.10+
- API Keys: Anthropic (wajib), OpenAI (opsional)
- RAM: ~3GB untuk model multilingual-e5-large (lokal)
- Internet: Dibutuhkan untuk download model HuggingFace (sekali saja)
