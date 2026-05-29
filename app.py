from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
CORS(app)

# ─── Load & Preprocessing ───────────────────────────────────────────────────
print("Memuat data berita...")

PATH_EXCEL = "berita_siap_search.xlsx"

df = pd.read_excel(PATH_EXCEL)

if 'Teks_Bersih' not in df.columns:
    print("Kolom 'Teks_Bersih' tidak ditemukan, menjalankan preprocessing...")

    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()

    processed = []
    for idx, row in df.iterrows():
        text = str(row['Isi Artikel'])
        text = text.lower()
        text = text.translate(dict((ord(c), None) for c in string.punctuation))
        text = stopword.remove(text)
        text = text.split()
        text = [stemmer.stem(w) for w in text]
        processed.append(' '.join(text))
        print(f"  preprocessing {idx+1}/{len(df)}...")

    df['Teks_Bersih'] = processed
    print("Preprocessing selesai!")
else:
    print("Kolom 'Teks_Bersih' ditemukan, skip preprocessing.")

processed_paper = df['Teks_Bersih'].tolist()
GROUND_TRUTH = {
    "uang": [0,2,5,6,7,11,14,24,28,29,47]
}
print(f"✅ {len(df)} berita siap digunakan!\n")

factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()
# ─── Evaluasi Precision, Recall, F1 ────────────────────────────────
def calculate_metrics(query_raw, query_tokens, retrieved_indices):

    query_key = query_raw.lower().strip()

    # ===== MODE 1 : GROUND TRUTH MANUAL =====
    if query_key in GROUND_TRUTH:

        relevant_set = set(GROUND_TRUTH[query_key])
        retrieved_set = set(retrieved_indices)

        tp = len(retrieved_set & relevant_set)
        fp = len(retrieved_set - relevant_set)
        fn = len(relevant_set - retrieved_set)

    # ===== MODE 2 : OTOMATIS (QUERY LAIN) =====
    else:

        relevant_indices = set()

        for idx, text in enumerate(processed_paper):
            words = set(str(text).split())

            if any(token in words for token in query_tokens):
                relevant_indices.add(idx)

        retrieved_set = set(retrieved_indices)

        tp = len(retrieved_set & relevant_indices)
        fp = len(retrieved_set - relevant_indices)
        fn = len(relevant_indices - retrieved_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

# ─── Endpoint Pencarian ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_file("google-merah.html")
@app.route('/search')
def search():
    query_raw = request.args.get('q', '').strip()
    top_n = int(request.args.get('n', 50))  # UBAH JADI 50 ATAU BERAPA AJA LU MAU

    if not query_raw:
        return jsonify({"error": "Query kosong"}), 400

    # Preprocessing query
    query = query_raw.lower()
    query = query.translate(dict((ord(c), None) for c in string.punctuation))
    query = stopword_remover.remove(query)
    query_tokens = [stemmer.stem(w) for w in query.split()]
    query_clean = ' '.join(query_tokens)

    if not query_clean.strip():
        return jsonify({"results": [], "total": 0, "query": query_raw})

    # Hitung TF-IDF + Cosine Similarity
    vectorizer = TfidfVectorizer(use_idf=True)
    all_texts = [query_clean] + processed_paper
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    q_vec = tfidf_matrix[0]
    scores = cosine_similarity(tfidf_matrix[1:], q_vec).flatten()

    # Urutkan berdasarkan skor tertinggi
    ranked_idx = np.argsort(-scores)

    # DEBUG - CEK DI TERMINAL
    print(f"Total dokumen: {len(scores)}")
    print(f"Top_n diminta: {top_n}")
    print(f"Score tertinggi 20: {scores[ranked_idx[:20]]}")

    results = []
    retrieved_indices = []
    for i, idx in enumerate(ranked_idx):  # LOOP SEMUA, TANPA BATASAN
        score = float(scores[idx])
        
        # SKIP YANG SCORE 0
        if score <= 0.0:
            continue
        retrieved_indices.append(idx)
        row = df.iloc[idx]
        results.append({
            "rank": len(results) + 1,
            "judul": str(row.get('Judul', '-')),
            "tanggal": str(row.get('Tanggal Rilis', '-')),
            "url": str(row.get('URL', '#')),
            "preview": str(row.get('Isi Artikel', ''))[:200] + '...',
            "score": round(score, 4)
        })
        
        # BERHENTI KALAU UDAH DAPAT top_n HASIL
        if len(results) >= top_n:
            break

        print(f"Total hasil dikembalikan: {len(results)}")

    metrics = calculate_metrics(
        query_raw,
        query_tokens,
        retrieved_indices
    )

    return jsonify({
        "results": results,
        "total": len(results),
        "query": query_raw,

        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],

        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"]
    })

# ─── Endpoint Saran ────────────────────────────────────────────────────────
@app.route('/suggest')
def suggest():
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2:
        return jsonify({"suggestions": []})

    suggestions = []
    for _, row in df.iterrows():
        judul = str(row.get('Judul', ''))
        if query in judul.lower():
            suggestions.append(judul)
        if len(suggestions) >= 6:
            break

    return jsonify({"suggestions": suggestions})


# ─── Jalankan Server ───────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)