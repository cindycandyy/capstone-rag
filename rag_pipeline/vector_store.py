"""
vector_store.py
---------------
Manajemen vector database menggunakan ChromaDB.

Modul ini menyediakan fungsi untuk:
  - Membuat atau memuat koleksi ChromaDB
  - Menambahkan dokumen (dengan chunking otomatis)
  - Melakukan pencarian cosine similarity

ChromaDB dipilih karena:
  - Setup mudah (pip install, tanpa Docker)
  - Persistent storage ke disk
  - Mendukung metadata filtering
  - Python SDK yang bersih dan idiomatis
"""

import os
import uuid
from typing import Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

from embedder import Embedder

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "indonesian_docs"


class VectorStore:
    """
    Wrapper ChromaDB untuk menyimpan dan mencari embedding dokumen.
    Mendukung chunking, penyimpanan metadata, dan cosine similarity search.
    """

    def __init__(self, embedder: Optional[Embedder] = None, persist_dir: str = None):
        """
        Inisialisasi ChromaDB client dan koleksi.

        Args:
            embedder: Instance Embedder. Jika None, buat baru dari env var.
            persist_dir: Path direktori penyimpanan ChromaDB.
        """
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.embedder = embedder or Embedder()

        # Buat direktori jika belum ada
        os.makedirs(self.persist_dir, exist_ok=True)

        # Inisialisasi ChromaDB dengan persistent storage
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Buat atau muat koleksi dengan cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Gunakan cosine similarity
        )

        print(
            f"[VectorStore] Koleksi '{COLLECTION_NAME}' siap. "
            f"Jumlah dokumen: {self.collection.count()}"
        )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> list[str]:
        """
        Memotong teks menjadi chunk-chunk dengan overlap.

        Menggunakan pemisahan berbasis karakter sebagai pendekatan ringan
        yang tidak memerlukan tokenizer eksternal untuk dokumen pendek.
        Untuk teks yang lebih panjang, overlap memastikan konteks tidak hilang
        di batas chunk.

        Args:
            text: Teks lengkap yang akan dipotong.
            chunk_size: Ukuran chunk dalam karakter (approx 300-500 tokens).
            overlap: Jumlah karakter overlap antar chunk.

        Returns:
            List string berisi chunk-chunk teks.
        """
        if len(text) <= chunk_size:
            return [text.strip()]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap  # Overlap antar chunk

        return chunks

    # ------------------------------------------------------------------
    # Inserting documents
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> int:
        """
        Tambahkan dokumen ke vector store (dengan chunking otomatis).

        Args:
            doc_id: Identifier unik untuk dokumen (misal: "doc1", "kominfo_001").
            text: Isi teks dokumen.
            metadata: Metadata tambahan (dict), misal: {"source": "kominfo", "category": "hoaks"}.
            chunk_size: Ukuran chunk dalam karakter.
            overlap: Overlap antar chunk dalam karakter.

        Returns:
            Jumlah chunk yang ditambahkan.
        """
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        metadata = metadata or {}

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk{i}"
            chunk_meta = {**metadata, "doc_id": doc_id, "chunk_index": i, "chunk_total": len(chunks)}

            documents.append(chunk)
            metadatas.append(chunk_meta)
            ids.append(chunk_id)

        # Embed semua chunk sekaligus (lebih efisien)
        embeddings = self.embedder.embed_batch(documents, is_query=False)

        # Tambahkan ke ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"[VectorStore] Dokumen '{doc_id}' ditambahkan sebagai {len(chunks)} chunk(s).")
        return len(chunks)

    def add_documents_bulk(self, documents: list[dict]) -> None:
        """
        Tambahkan banyak dokumen sekaligus.

        Args:
            documents: List dict dengan format:
                {
                    "id": "doc1",
                    "text": "isi dokumen...",
                    "metadata": {"source": "kominfo", "category": "hoaks"}  # opsional
                }
        """
        for doc in documents:
            self.add_document(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Cari dokumen yang paling relevan dengan query menggunakan cosine similarity.

        Args:
            query: Teks query yang akan dicari.
            top_k: Jumlah hasil teratas yang dikembalikan.
            where: Filter metadata ChromaDB (opsional).
                   Contoh: {"category": "hoaks"} atau {"source": "kominfo"}

        Returns:
            List dict dengan format:
            [
                {
                    "id": "doc1_chunk0",
                    "text": "teks chunk...",
                    "metadata": {...},
                    "distance": 0.12,   # jarak cosine (lebih kecil = lebih mirip)
                    "score": 0.88,      # similarity score (1 - distance)
                }
            ]
        """
        # Embed query
        query_embedding = self.embedder.embed(query, is_query=True)

        # Hitung jumlah dokumen yang tersedia
        total_docs = self.collection.count()
        if total_docs == 0:
            print("[VectorStore] Koleksi kosong. Tambahkan dokumen terlebih dahulu.")
            return []

        effective_k = min(top_k, total_docs)

        # Query ke ChromaDB
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Format hasil
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1.0 - results["distances"][0][i],  # Similarity score
            })

        return output

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Kembalikan jumlah total chunk dalam koleksi."""
        return self.collection.count()

    def reset(self) -> None:
        """Hapus semua dokumen dari koleksi (untuk testing)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorStore] Koleksi '{COLLECTION_NAME}' direset.")
