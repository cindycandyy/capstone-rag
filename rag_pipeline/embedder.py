"""
embedder.py
-----------
Embedding model wrapper yang mendukung pergantian model via konfigurasi .env.

Mendukung 3 model:
  - "openai"           → text-embedding-3-large (via OpenAI API)
  - "labse"            → LaBSE (via sentence-transformers, lokal)
  - "multilingual-e5"  → multilingual-e5-large (via sentence-transformers, lokal)

Cara Penggunaan:
    from embedder import Embedder
    embedder = Embedder()  # reads EMBEDDING_MODEL from .env
    vector = embedder.embed("teks dalam bahasa indonesia")
    vectors = embedder.embed_batch(["kalimat 1", "kalimat 2"])
"""

import os
from typing import Union
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multilingual-e5").lower()


class Embedder:
    """
    Wrapper yang menyediakan antarmuka tunggal untuk berbagai embedding model.
    Model dipilih via environment variable EMBEDDING_MODEL.
    """

    SUPPORTED_MODELS = ["openai", "labse", "multilingual-e5"]

    def __init__(self, model_name: str = None):
        """
        Inisialisasi embedder dengan model yang dipilih.

        Args:
            model_name: Override model dari env var. Pilihan: "openai", "labse",
                        "multilingual-e5". Jika None, ambil dari EMBEDDING_MODEL env var.
        """
        self.model_name = (model_name or EMBEDDING_MODEL).lower()

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' tidak dikenal. "
                f"Pilihan yang tersedia: {self.SUPPORTED_MODELS}"
            )

        # Lazy-load: model diinisialisasi sekali saat pertama dibutuhkan
        self._client = None
        self._st_model = None

        print(f"[Embedder] Menggunakan model: '{self.model_name}'")

    # ------------------------------------------------------------------
    # Private: model initialization
    # ------------------------------------------------------------------

    def _get_openai_client(self):
        """Inisialisasi OpenAI client (lazy-load)."""
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY tidak ditemukan di .env. "
                    "Tambahkan key Anda di file .env."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _get_st_model(self):
        """Inisialisasi sentence-transformers model (lazy-load)."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            model_map = {
                "labse": "sentence-transformers/LaBSE",
                "multilingual-e5": "intfloat/multilingual-e5-large",
            }
            hf_model_id = model_map[self.model_name]
            print(f"[Embedder] Memuat model lokal: {hf_model_id} ...")
            self._st_model = SentenceTransformer(hf_model_id)
            print(f"[Embedder] Model berhasil dimuat.")
        return self._st_model

    # ------------------------------------------------------------------
    # Private: prefix helper untuk E5
    # ------------------------------------------------------------------

    def _apply_e5_prefix(self, texts: list[str], is_query: bool) -> list[str]:
        """
        multilingual-e5-large memerlukan prefix khusus:
          - "query: " untuk query
          - "passage: " untuk dokumen/chunk
        """
        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Jumlah dimensi vektor untuk model yang dipilih."""
        dim_map = {
            "openai": 3072,
            "labse": 768,
            "multilingual-e5": 1024,
        }
        return dim_map[self.model_name]

    def embed(self, text: str, is_query: bool = False) -> list[float]:
        """
        Embed satu teks menjadi vektor.

        Args:
            text: Teks yang akan di-embed.
            is_query: True jika teks adalah query (relevan untuk E5 prefix).

        Returns:
            List float representing the embedding vector.
        """
        results = self.embed_batch([text], is_query=is_query)
        return results[0]

    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """
        Embed sekumpulan teks sekaligus (lebih efisien daripada satu per satu).

        Args:
            texts: List teks yang akan di-embed.
            is_query: True jika semua teks adalah query.

        Returns:
            List of embedding vectors (list of lists of floats).
        """
        if not texts:
            return []

        if self.model_name == "openai":
            return self._embed_openai(texts)
        elif self.model_name in ("labse", "multilingual-e5"):
            return self._embed_sentence_transformer(texts, is_query=is_query)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed menggunakan OpenAI text-embedding-3-large."""
        client = self._get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts,
        )
        # Kembalikan vektor dalam urutan yang sama dengan input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def _embed_sentence_transformer(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Embed menggunakan sentence-transformers (LaBSE atau E5)."""
        model = self._get_st_model()

        # Terapkan prefix E5 jika model adalah multilingual-e5
        if self.model_name == "multilingual-e5":
            texts = self._apply_e5_prefix(texts, is_query=is_query)

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,  # Normalisasi untuk cosine similarity
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings.tolist()
