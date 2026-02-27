"""
retriever.py
------------
Modul retrieval: menghubungkan query ke vector store dan mengembalikan
chunk-chunk dokumen yang paling relevan.

Alur: query → embed → cosine search → return top-K chunks
"""

import os
from typing import Optional
from dotenv import load_dotenv

from embedder import Embedder
from vector_store import VectorStore

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "5"))


class Retriever:
    """
    Mengelola proses retrieval dari vector store.
    Menerima query teks, mencari chunk yang relevan, dan mengembalikannya
    dalam format terstruktur untuk diinjeksikan ke prompt.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        top_k: int = None,
    ):
        """
        Inisialisasi Retriever.

        Args:
            vector_store: Instance VectorStore. Jika None, buat baru.
            embedder: Instance Embedder yang sama dengan VectorStore.
            top_k: Jumlah dokumen teratas yang diambil.
        """
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore(embedder=self.embedder)
        self.top_k = top_k or TOP_K

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Ambil chunk dokumen yang paling relevan untuk query yang diberikan.

        Args:
            query: Teks query/klaim yang ingin dianalisis.
            top_k: Override jumlah hasil (default: dari env TOP_K).
            where: Filter metadata opsional untuk mempersempit pencarian.

        Returns:
            List dict berisi chunk yang relevan:
            [
                {
                    "id": "doc1_chunk0",
                    "text": "isi chunk...",
                    "metadata": {"doc_id": "doc1", "source": "kominfo", ...},
                    "score": 0.85,
                }
            ]
        """
        k = top_k or self.top_k

        print(f"[Retriever] Query: '{query}'")
        print(f"[Retriever] Mencari top-{k} chunk yang relevan...")

        results = self.vector_store.search(query=query, top_k=k, where=where)

        print(f"[Retriever] Ditemukan {len(results)} chunk:")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']:.4f} | id={r['id']}")

        return results

    def retrieve_with_threshold(
        self,
        query: str,
        min_score: float = 0.3,
        top_k: int = None,
    ) -> list[dict]:
        """
        Ambil chunk yang relevan dengan filter minimum similarity score.

        Berguna untuk menghindari injeksi konteks yang tidak relevan ke prompt.

        Args:
            query: Teks query.
            min_score: Score minimum (0–1). Chunk di bawah threshold diabaikan.
            top_k: Jumlah kandidat awal sebelum difilter.

        Returns:
            List chunk yang melewati threshold similarity.
        """
        candidates = self.retrieve(query=query, top_k=top_k)
        filtered = [r for r in candidates if r["score"] >= min_score]

        print(
            f"[Retriever] Setelah filter threshold={min_score}: "
            f"{len(filtered)}/{len(candidates)} chunk dipertahankan."
        )
        return filtered
