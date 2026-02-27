"""
benchmark_embeddings.py
-----------------------
Skrip standalone untuk membandingkan performa 3 embedding model pada teks Bahasa Indonesia.

Benchmark mencakup:
  1. Kecepatan embedding (100 kalimat sampel)
  2. Tes semantic similarity (5 pasang kalimat similar vs dissimilar)
  3. Tabel perbandingan akhir

Jalankan dengan:
    python benchmark_embeddings.py

Catatan:
  - Model LaBSE dan multilingual-e5 akan diunduh secara otomatis (~500MB per model)
  - Pastikan OPENAI_API_KEY tersedia di .env untuk menguji model OpenAI
"""

import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 100 Kalimat Bahasa Indonesia untuk benchmark kecepatan
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_SENTENCES = [
    # Hoaks & Disinformasi
    "Vaksin COVID-19 mengandung microchip 5G yang dapat melacak manusia.",
    "Virus corona diciptakan oleh laboratorium di Wuhan.",
    "Minum air panas dapat membunuh virus corona.",
    "Bawang putih bisa menyembuhkan COVID-19.",
    "Bill Gates menggunakan vaksin untuk mengurangi populasi dunia.",
    # Ujaran Kebencian
    "Kelompok etnis tertentu harus diusir dari Indonesia.",
    "Suku X adalah sumber masalah ekonomi negara.",
    "Umat agama Y tidak layak tinggal di Indonesia.",
    "Perempuan tidak seharusnya memimpin perusahaan.",
    "Kaum difabel tidak produktif untuk masyarakat.",
    # Konten Netral
    "Jakarta adalah ibu kota Indonesia yang terletak di Pulau Jawa.",
    "Badan Pusat Statistik mencatat pertumbuhan ekonomi 5,3 persen.",
    "Presiden Joko Widodo meresmikan jalan tol baru di Sumatera.",
    "Sepak bola adalah olahraga paling populer di Indonesia.",
    "Reog Ponorogo adalah kesenian tradisional dari Jawa Timur.",
    # Definisi Hukum
    "UU ITE Pasal 28 mengatur penyebaran informasi yang meresahkan masyarakat.",
    "KUHP Pasal 310 mendefinisikan penghinaan dan pencemaran nama baik.",
    "Komnas HAM bertugas memantau pelanggaran hak asasi manusia di Indonesia.",
    "Mahkamah Konstitusi menguji konstitusionalitas undang-undang.",
    "Ombudsman menerima pengaduan malpraktek pelayanan publik.",
    # Fact-check
    "MAFINDO telah membantah klaim yang menyebutkan 5G berbahaya bagi kesehatan.",
    "Kominfo mengverifikasi bahwa foto tersebut adalah manipulasi digital.",
    "Cek Fakta Kompas menyimpulkan berita tersebut adalah hoaks.",
    "WHO menegaskan tidak ada bukti ilmiah bahwa kejus lemon menyembuhkan kanker.",
    "Reuters Fact Check mengkonfirmasi berita palsu tentang vaksin polio.",
    # Kalimat campuran panjang
    "Berdasarkan penelitian ilmiah yang diterbitkan di jurnal Nature, vaksin mRNA aman.",
    "Kominfo bekerja sama dengan platform media sosial untuk menghapus konten hoaks.",
    "Pendidikan literasi digital penting untuk mencegah penyebaran disinformasi.",
    "Generasi muda harus kritis dalam menyaring informasi dari media sosial.",
    "Algoritma media sosial sering kali memperkuat gelembung informasi pengguna.",
    # Topik lain
    "Perubahan iklim mengancam ketahanan pangan Indonesia di masa depan.",
    "Deforestasi di Kalimantan menyebabkan kepunahan orangutan.",
    "Banjir Jakarta terjadi setiap tahun akibat buruknya sistem drainase.",
    "BPJS Kesehatan memberikan layanan kesehatan bagi seluruh warga Indonesia.",
    "Kurikulum Merdeka dirancang untuk meningkatkan kualitas pendidikan nasional.",
    # Versi pendek
    "Hoaks.", "Disinformasi.", "Fitnah.", "Ujaran kebencian.", "Fakta terverifikasi.",
    "Berita palsu.", "Klaim tidak berdasar.", "Informasi menyesatkan.", "Propaganda.",
    "Manipulasi media.", "Cek fakta.", "Konfirmasi sumber.", "Data valid.", "Referensi.",
    "Klarifikasi resmi.", "Bantahan ilmiah.", "Verifikasi independen.", "Audit fakta.",
    "Investigasi jurnalistik.", "Laporan resmi pemerintah.",
    # Campuran formal dan informal
    "Katanya vaksin bikin mandul, tapi itu hoaks udah dibantah dokter.",
    "Gue denger ada yang bilang 5G nyebarin virus, tapi itu mah ngawur.",
    "Jangan percaya berita dari grup WhatsApp tanpa cek dulu ke sumber resmi.",
    "Berita yang viral di TikTok soal obat COVID ternyata sudah terbukti hoaks.",
    "Hati-hati sama konten yang provokatif dan ngajak benci sama orang lain.",
    # Lebih banyak untuk mencapai 100
    "Kebebasan berekspresi dijamin UUD 1945 Pasal 28.",
    "Media massa bertanggung jawab menyajikan informasi yang akurat.",
    "Literasi media adalah kemampuan kritis membaca informasi.",
    "Algoritma rekomendasi memperkuat echo chamber di platform digital.",
    "Regulasi konten digital di Indonesia terus berkembang.",
    "Kepolisian menangkap penyebar hoaks yang memprovokasi konflik SARA.",
    "Masyarakat diminta melapor ke Kominfo jika menemukan konten berbahaya.",
    "Tokoh agama berperan penting dalam mencegah radikalisasi online.",
    "Pendidikan karakter di sekolah mencakup literasi digital dan kritis.",
    "Berita bohong dapat dikenakan sanksi pidana sesuai UU ITE.",
    "Platform media sosial wajib memenuhi regulasi takedown konten ilegal.",
    "Koalisi masyarakat sipil mendorong transparansi algoritma platform digital.",
    "Jurnalisme data semakin penting dalam era informasi yang kompleks.",
    "Privasi data pengguna dilindungi oleh UU Perlindungan Data Pribadi.",
    "Pemerintah meluncurkan portal cek fakta resmi di website Kominfo.",
    "Komunitas fact-checker independen berperan dalam menangkal hoaks.",
    "Konten deepfake semakin sulit dibedakan dari video asli.",
    "Teknik OSINT digunakan untuk memverifikasi klaim di media sosial.",
    "Kecerdasan buatan dapat membantu deteksi disinformasi secara otomatis.",
    "Narasi disinformasi sering memanfaatkan ketidakpastian untuk menyebar.",
    "Bot media sosial digunakan untuk amplifikasi konten propaganda.",
    "Koordinasi internasional diperlukan untuk menangani disinformasi lintas batas.",
    "Transparansi iklan politik di platform digital menjadi isu penting.",
    "Kampanye hitam dalam pemilu sering menggunakan disinformasi sebagai senjata.",
    "PemVerifikasi fakta harus independen dari tekanan politik dan bisnis.",
    "Standar internasional verifikasi fakta dikembangkan oleh IFCN.",
    "Pelatihan jurnalis dalam teknik verifikasi fakta sangat dibutuhkan.",
    "Kolaborasi lintas newsroom memperkuat upaya debunking disinformasi.",
    "Media komunitas lokal berperan dalam melawan disinformasi di daerah.",
    "Pendekatan prebunking efektif mencegah penyebaran hoaks sebelum viral.",
    "Inokulasi informasi membantu publik mengenali taktik manipulasi.",
    "Program literasi media harus menjangkau seluruh lapisan masyarakat.",
    "Orang tua perlu dibekali kemampuan mendampingi anak berselancar internet.",
    "Generasi Z cenderung lebih kritis terhadap konten yang mereka konsumsi.",
    "Influencer media sosial bertanggung jawab atas konten yang mereka sebarkan.",
    "Moderasi konten memerlukan keseimbangan antara kebebasan dan tanggung jawab.",
    "Pelaporan konten berbahaya harus mudah dan dapat diakses semua pengguna.",
    "Sistem peringatan dini disinformasi dapat mencegah krisis informasi.",
    "Kerja sama pemerintah dan masyarakat sipil kunci tangkal hoaks.",
    "Bahasa Indonesia memiliki keunikan yang perlu dipahami sistem NLP.",
    "Model NLP multibahasa mampu memproses teks Bahasa Indonesia dengan baik.",
    "Embedding model yang baik harus menangkap nuansa bahasa lokal.",
]

# Pastikan tepat 100 kalimat
BENCHMARK_SENTENCES = BENCHMARK_SENTENCES[:100]


# ─────────────────────────────────────────────────────────────────────────────
# 5 Pasang Kalimat untuk Semantic Similarity Test
# ─────────────────────────────────────────────────────────────────────────────
SIMILARITY_PAIRS = [
    {
        "label": "Pair 1 (Similar) — konteks hoaks vaksin",
        "sentence_a": "Vaksin COVID-19 mengandung microchip 5G yang dapat melacak manusia.",
        "sentence_b": "Klaim chip 5G dalam vaksin adalah hoaks yang telah dibantah secara ilmiah oleh Kominfo.",
        "expected": "SIMILAR",
    },
    {
        "label": "Pair 2 (Similar) — ujaran kebencian SARA",
        "sentence_a": "Ujaran kebencian terhadap suku dan agama minoritas di media sosial.",
        "sentence_b": "Hate speech berbasis SARA dilarang oleh UU ITE Pasal 28 ayat 2.",
        "expected": "SIMILAR",
    },
    {
        "label": "Pair 3 (Similar) — defamasi/fitnah",
        "sentence_a": "Fitnah adalah tuduhan palsu yang bertujuan merusak reputasi seseorang.",
        "sentence_b": "Defamasi atau pencemaran nama baik diatur dalam KUHP Pasal 310.",
        "expected": "SIMILAR",
    },
    {
        "label": "Pair 4 (Dissimilar) — topik berbeda",
        "sentence_a": "Harga beras naik 30 persen di pasar tradisional Jawa Tengah bulan ini.",
        "sentence_b": "Vaksin mengandung zat kimia berbahaya yang merusak sistem imun.",
        "expected": "DISSIMILAR",
    },
    {
        "label": "Pair 5 (Dissimilar) — sangat tidak relevan",
        "sentence_a": "Cuaca cerah di Jakarta hari ini dengan suhu 32 derajat Celsius.",
        "sentence_b": "Hoaks menyebar di WhatsApp tentang obat COVID dari tanaman herbal.",
        "expected": "DISSIMILAR",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Hitung cosine similarity antara dua vektor."""
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def benchmark_speed(model_name: str) -> dict:
    """
    Benchmark kecepatan embedding 100 kalimat untuk model tertentu.

    Returns:
        Dict berisi hasil benchmark (waktu, kecepatan, dimensi).
    """
    from embedder import Embedder

    print(f"\n{'─' * 50}")
    print(f"🚀 Benchmarking: {model_name}")
    print(f"{'─' * 50}")

    embedder = Embedder(model_name=model_name)

    print(f"  Dimensi vektor: {embedder.dimensions}")

    # Warmup (inisialisasi model)
    print("  Warmup embedding...")
    _ = embedder.embed_batch(BENCHMARK_SENTENCES[:2], is_query=False)

    # Actual benchmark
    print(f"  Embedding {len(BENCHMARK_SENTENCES)} kalimat...")
    start_time = time.time()
    embeddings = embedder.embed_batch(BENCHMARK_SENTENCES, is_query=False)
    elapsed = time.time() - start_time

    speed = len(BENCHMARK_SENTENCES) / elapsed
    print(f"  ✅ Selesai dalam {elapsed:.2f}s → {speed:.1f} kalimat/detik")

    return {
        "model": model_name,
        "dimensions": embedder.dimensions,
        "n_sentences": len(BENCHMARK_SENTENCES),
        "elapsed_seconds": round(elapsed, 2),
        "sentences_per_second": round(speed, 1),
        "embeddings": embeddings,  # Simpan untuk tes similarity
    }


def test_semantic_similarity(model_name: str, embedder_ref=None) -> list[dict]:
    """
    Uji semantic similarity untuk 5 pasang kalimat.

    Args:
        model_name: Nama model embedding.
        embedder_ref: Instance Embedder yang sudah ada (untuk menghindari reload model).

    Returns:
        List dict berisi hasil similarity untuk setiap pasang kalimat.
    """
    from embedder import Embedder

    embedder = embedder_ref or Embedder(model_name=model_name)
    results = []

    print(f"\n  📐 Semantic Similarity Test — {model_name}")
    for pair in SIMILARITY_PAIRS:
        vec_a = embedder.embed(pair["sentence_a"], is_query=False)
        vec_b = embedder.embed(pair["sentence_b"], is_query=False)
        score = cosine_similarity(vec_a, vec_b)

        results.append({
            "label": pair["label"],
            "sentence_a": pair["sentence_a"][:60] + "...",
            "sentence_b": pair["sentence_b"][:60] + "...",
            "expected": pair["expected"],
            "cosine_score": round(score, 4),
        })
        status = "✅" if (
            (pair["expected"] == "SIMILAR" and score > 0.5) or
            (pair["expected"] == "DISSIMILAR" and score < 0.5)
        ) else "❌"
        print(f"    {status} {pair['label']}: {score:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("📊 EMBEDDING MODEL BENCHMARK")
    print("   Bahasa Indonesia — Disinformasi & Ujaran Kebencian")
    print("=" * 60)

    models_to_test = []

    # Cek apakah OpenAI key tersedia
    if os.getenv("OPENAI_API_KEY"):
        models_to_test.append("openai")
    else:
        print("\n⚠️  OPENAI_API_KEY tidak ditemukan — melewati model OpenAI.")

    models_to_test.extend(["labse", "multilingual-e5"])

    speed_results = []
    similarity_results = {}

    # ── Benchmark setiap model ──────────────────────────────────────
    for model_name in models_to_test:
        try:
            speed_data = benchmark_speed(model_name)
            speed_results.append({
                "model": model_name,
                "dimensions": speed_data["dimensions"],
                "time_100s": speed_data["elapsed_seconds"],
                "speed_s_per_sec": speed_data["sentences_per_second"],
            })

            sim_results = test_semantic_similarity(model_name)
            similarity_results[model_name] = sim_results

        except Exception as e:
            print(f"\n❌ Error saat benchmark {model_name}: {e}")
            speed_results.append({
                "model": model_name,
                "dimensions": "N/A",
                "time_100s": "ERROR",
                "speed_s_per_sec": "ERROR",
            })

    # ── Tampilkan Tabel Kecepatan ───────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📈 HASIL BENCHMARK KECEPATAN (100 kalimat)")
    print("=" * 60)

    try:
        from tabulate import tabulate
        print(tabulate(
            speed_results,
            headers={"model": "Model", "dimensions": "Dimensi",
                     "time_100s": "Waktu (detik)", "speed_s_per_sec": "Kalimat/detik"},
            tablefmt="rounded_grid",
        ))
    except ImportError:
        for r in speed_results:
            print(f"  {r['model']:20s} | dim={r['dimensions']} | "
                  f"time={r['time_100s']}s | speed={r['speed_s_per_sec']}/s")

    # ── Tampilkan Tabel Semantic Similarity ────────────────────────
    print("\n\n" + "=" * 60)
    print("🧪 HASIL SEMANTIC SIMILARITY TEST")
    print("=" * 60)

    for model_name, pairs in similarity_results.items():
        print(f"\nModel: {model_name}")
        try:
            from tabulate import tabulate
            table_data = [
                {
                    "Pair": p["label"],
                    "Expected": p["expected"],
                    "Score": p["cosine_score"],
                    "Result": "✅ PASS" if (
                        (p["expected"] == "SIMILAR" and p["cosine_score"] > 0.5) or
                        (p["expected"] == "DISSIMILAR" and p["cosine_score"] < 0.5)
                    ) else "❌ FAIL",
                }
                for p in pairs
            ]
            print(tabulate(table_data, headers="keys", tablefmt="rounded_grid"))
        except ImportError:
            for p in pairs:
                print(f"  {p['label']}: {p['cosine_score']:.4f} ({p['expected']})")

    print("\n\n✅ Benchmark selesai!")
    print("   Lihat docs/embedding_research.md untuk analisis lengkap dan rekomendasi.")


if __name__ == "__main__":
    main()
