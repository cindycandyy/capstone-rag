"""
prompt_builder.py
-----------------
Membangun prompt yang diinjeksikan dengan konteks dari retrieved chunks.

Modul ini bertanggung jawab untuk:
  1. Mengambil chunk-chunk yang relevan dari retriever
  2. Memformatnya menjadi blok konteks yang terstruktur
  3. Menggabungkannya dalam template prompt yang sesuai untuk analisis
     disinformasi, defamasi, dan ujaran kebencian dalam konteks Bahasa Indonesia
"""

SYSTEM_PROMPT = """Anda adalah analis konten ahli yang bertugas mendeteksi disinformasi, 
defamasi, dan ujaran kebencian dalam Bahasa Indonesia.

Tugas Anda adalah menganalisis teks yang diberikan berdasarkan konteks referensi yang 
telah diambil dari database dokumen resmi (fact-check, regulasi, definisi hukum).

Berikan analisis yang akurat, objektif, dan berbasis bukti. Selalu dasarkan kesimpulan 
Anda pada konteks referensi yang diberikan, bukan pada asumsi.

Format jawaban Anda HARUS mengikuti struktur JSON berikut (tanpa markdown di luar JSON):
{
  "classification": "<DISINFORMATION | DEFAMATION | HATE_SPEECH | CLEAN | MULTIPLE>",
  "confidence": "<HIGH | MEDIUM | LOW>",
  "confidence_score": <float 0.0-1.0>,
  "reasoning": "<penjelasan detail mengapa konten diklasifikasikan demikian>",
  "evidence": ["<kutipan bukti 1 dari konteks>", "<kutipan bukti 2>"],
  "sources_used": ["<id dokumen referensi yang relevan>"],
  "recommendation": "<rekomendasi tindakan atau klarifikasi>"
}"""


ANALYSIS_PROMPT_TEMPLATE = """## Konteks Referensi

Berikut adalah {n_chunks} dokumen referensi yang paling relevan dengan konten yang dianalisis:

{context_block}

---

## Konten yang Dianalisis

```
{query}
```

---

## Instruksi Analisis

Berdasarkan konteks referensi di atas, analisis apakah konten tersebut mengandung:
- **DISINFORMATION**: Informasi yang salah, menyesatkan, atau merupakan hoaks yang telah dibantah
- **DEFAMATION (Defamasi/Fitnah)**: Tuduhan atau pernyataan yang menyerang kehormatan seseorang tanpa bukti
- **HATE_SPEECH (Ujaran Kebencian)**: Konten yang menimbulkan kebencian terhadap individu/kelompok berdasarkan SARA
- **CLEAN**: Konten yang tidak mengandung unsur di atas
- **MULTIPLE**: Jika mengandung lebih dari satu kategori di atas

Berikan analisis lengkap dalam format JSON yang telah ditentukan."""


def build_context_block(retrieved_chunks: list[dict]) -> str:
    """
    Memformat chunk-chunk yang diperoleh menjadi blok teks terstruktur.

    Setiap chunk diformat dengan:
      - Nomor urut
      - ID dokumen dan metadata sumber
      - Similarity score
      - Isi teks chunk

    Args:
        retrieved_chunks: List dict dari retriever, masing-masing berisi
                          "id", "text", "metadata", "score".

    Returns:
        String berisi blok konteks terformat.
    """
    if not retrieved_chunks:
        return "(Tidak ada dokumen referensi yang relevan ditemukan)"

    lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        doc_id = chunk["metadata"].get("doc_id", chunk["id"])
        source = chunk["metadata"].get("source", "unknown")
        category = chunk["metadata"].get("category", "general")
        score = chunk.get("score", 0.0)

        lines.append(f"### [{i}] Sumber: {doc_id}")
        lines.append(f"**Kategori**: {category} | **Relevansi**: {score:.2%}")
        if source != "unknown":
            lines.append(f"**Asal**: {source}")
        lines.append("")
        lines.append(f"> {chunk['text']}")
        lines.append("")

    return "\n".join(lines)


def build_prompt(query: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
    """
    Bangun system prompt dan user prompt lengkap dengan konteks yang diinjeksikan.

    Args:
        query: Teks/klaim yang ingin dianalisis.
        retrieved_chunks: Hasil dari retriever (list of chunk dicts).

    Returns:
        Tuple (system_prompt, user_prompt) siap dikirim ke LLM.
    """
    context_block = build_context_block(retrieved_chunks)

    user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        n_chunks=len(retrieved_chunks),
        context_block=context_block,
        query=query,
    )

    return SYSTEM_PROMPT, user_prompt


def build_prompt_messages(query: str, retrieved_chunks: list[dict]) -> list[dict]:
    """
    Bangun daftar pesan dalam format Anthropic/OpenAI untuk API call.

    Args:
        query: Teks yang dianalisis.
        retrieved_chunks: Chunk dokumen yang diambil oleh retriever.

    Returns:
        List dict siap dikirim sebagai `messages` ke LLM API.
    """
    system_prompt, user_prompt = build_prompt(query, retrieved_chunks)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
