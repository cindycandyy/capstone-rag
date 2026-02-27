"""
main.py
-------
Entry point untuk Agentic RAG Pipeline deteksi disinformasi Bahasa Indonesia.

Menjalankan pipeline lengkap:
  1. Load dokumen referensi ke vector store
  2. Terima query input
  3. Retrieve chunk relevan (cosine similarity search)
  4. Bangun prompt dengan konteks yang diinjeksikan
  5. Generate analisis menggunakan Claude
  6. Tampilkan hasil terstruktur

Jalankan dengan:
    python main.py
Atau jalankan dengan query kustom:
    python main.py --query "teks yang ingin dianalisis"
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Dokumen referensi mock (5 dokumen sampel)
# ─────────────────────────────────────────────
REFERENCE_DOCUMENTS = [
    {
        "id": "doc1",
        "text": (
            "Berdasarkan fact-check Kominfo, klaim bahwa vaksin mengandung chip 5G "
            "adalah hoaks yang telah dibantah secara ilmiah."
        ),
        "metadata": {
            "source": "kominfo",
            "category": "hoaks",
            "topic": "vaksin",
            "language": "id",
        },
    },
    {
        "id": "doc2",
        "text": (
            "Ujaran kebencian berdasarkan UU ITE Pasal 28 ayat 2 mencakup penyebaran "
            "informasi yang menimbulkan rasa kebencian terhadap individu atau kelompok "
            "berdasarkan SARA."
        ),
        "metadata": {
            "source": "uu_ite",
            "category": "hate_speech",
            "topic": "regulasi",
            "language": "id",
        },
    },
    {
        "id": "doc3",
        "text": (
            "Fitnah dalam hukum Indonesia diatur dalam KUHP Pasal 310, yaitu menyerang "
            "kehormatan seseorang dengan tuduhan tanpa bukti."
        ),
        "metadata": {
            "source": "kuhp",
            "category": "defamasi",
            "topic": "hukum_pidana",
            "language": "id",
        },
    },
    {
        "id": "doc4",
        "text": (
            "Hoaks COVID-19 yang menyebar di WhatsApp pada 2021 termasuk klaim bahwa "
            "minum air panas setiap 15 menit dapat membunuh virus corona."
        ),
        "metadata": {
            "source": "mafindo",
            "category": "hoaks",
            "topic": "covid19",
            "language": "id",
        },
    },
    {
        "id": "doc5",
        "text": (
            "Turn Back Hoax MAFINDO mendefinisikan disinformasi sebagai informasi yang "
            "salah dan disebarkan dengan niat untuk menipu."
        ),
        "metadata": {
            "source": "mafindo",
            "category": "disinformasi",
            "topic": "definisi",
            "language": "id",
        },
    },
]

# Query default untuk pengujian awal
DEFAULT_QUERY = "apakah klaim vaksin mengandung microchip itu benar?"


def setup_vector_store(force_reload: bool = False):
    """
    Inisialisasi vector store dan muat dokumen referensi.

    Jika dokumen sudah ada di ChromaDB (dari run sebelumnya), tidak perlu
    dimuat ulang kecuali force_reload=True.

    Args:
        force_reload: Jika True, hapus koleksi lama dan muat ulang semua dokumen.

    Returns:
        Instance VectorStore yang sudah berisi dokumen referensi.
    """
    from embedder import Embedder
    from vector_store import VectorStore

    print("\n" + "=" * 60)
    print("🔧 SETUP VECTOR STORE")
    print("=" * 60)

    embedder = Embedder()
    vs = VectorStore(embedder=embedder)

    if force_reload and vs.count() > 0:
        print("[Main] Force reload: menghapus koleksi lama...")
        vs.reset()

    if vs.count() == 0:
        print(f"[Main] Memuat {len(REFERENCE_DOCUMENTS)} dokumen referensi ke ChromaDB...")
        vs.add_documents_bulk(REFERENCE_DOCUMENTS)
        print(f"[Main] ✅ {vs.count()} chunk berhasil disimpan ke vector store.")
    else:
        print(f"[Main] ✅ Vector store sudah berisi {vs.count()} chunk. Melewati indexing.")

    return embedder, vs


def run_pipeline(query: str, force_reload: bool = False) -> dict:
    """
    Jalankan pipeline RAG end-to-end untuk query yang diberikan.

    Alur lengkap:
      Query → Embed → Search → Retrieve → Inject to Prompt → Generate → Return

    Args:
        query: Teks/klaim yang ingin dianalisis.
        force_reload: Paksa reload dokumen ke vector store.

    Returns:
        Dict berisi hasil analisis lengkap.
    """
    from retriever import Retriever
    from prompt_builder import build_prompt
    from generator import Generator

    # ── Step 1: Setup vector store ──────────────────────────────────
    embedder, vs = setup_vector_store(force_reload=force_reload)

    # ── Step 2: Retrieve dokumen relevan ────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 RETRIEVAL")
    print("=" * 60)

    retriever = Retriever(vector_store=vs, embedder=embedder)
    retrieved_chunks = retriever.retrieve(query=query)

    # ── Step 3: Bangun prompt dengan konteks ────────────────────────
    print("\n" + "=" * 60)
    print("📝 PROMPT BUILDING")
    print("=" * 60)

    system_prompt, user_prompt = build_prompt(query, retrieved_chunks)
    print(f"[Main] Prompt dibangun dengan {len(retrieved_chunks)} chunk konteks.")

    # ── Step 4: Generate analisis ───────────────────────────────────
    print("\n" + "=" * 60)
    print("🤖 GENERATING ANALYSIS (Claude)")
    print("=" * 60)

    generator = Generator()
    result = generator.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    # ── Step 5: Tampilkan hasil ─────────────────────────────────────
    print("\n")
    print(result.display())

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "analysis": result.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Deteksi Disinformasi Bahasa Indonesia"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Teks atau klaim yang ingin dianalisis",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Paksa reload dokumen ke vector store (hapus data lama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override embedding model: 'openai', 'labse', 'multilingual-e5'",
    )

    args = parser.parse_args()

    # Override embedding model jika diberikan via CLI
    if args.model:
        os.environ["EMBEDDING_MODEL"] = args.model

    # Enable utf-8 encoding for standard output on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 60)
    print("🛡️  AGENTIC RAG - DETEKSI DISINFORMASI BAHASA INDONESIA")
    print("=" * 60)
    print(f"Query: {args.query}")

    result = run_pipeline(query=args.query, force_reload=args.reload)

    # Return exit code berdasarkan hasil klasifikasi
    classification = result["analysis"].get("classification", "UNKNOWN")
    if classification in ("ERROR", "PARSE_ERROR", "UNKNOWN"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
