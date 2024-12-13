# Anggota kelompok
# Latif Ardiansyah 22.12.2599
# Reyhan Dwi Wira Allofadieka 22.12.2563

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Cek apakah file ada dan kolom 'judul_prosessing' tersedia
try:
    jurnal_df = pd.read_excel('jurnal_sinta.xlsx')
    if 'judul_prosessing' not in jurnal_df.columns:
        raise ValueError("Kolom 'judul_prosessing' tidak ditemukan di file 'jurnal_sinta.xlsx'.")
except FileNotFoundError:
    raise FileNotFoundError("File 'jurnal_sinta.xlsx' tidak ditemukan. Pastikan file ada di direktori yang benar.")
except Exception as e:
    raise Exception(f"Error saat membaca file: {e}")

# Hapus data kosong
jurnal_df = jurnal_df[jurnal_df['judul_prosessing'].notnull()]

# Inisialisasi pembersihan teks
clean_spcl = re.compile(r'[/(){}\[\]\|@,;]')
clean_symbol = re.compile(r'[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
stopwords = sastrawi.get_stop_words()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi pembersihan teks
def clean_text(text):
    """Membersihkan teks dengan menghapus simbol, stopwords, dan stemming."""
    text = text.lower()
    text = clean_spcl.sub(' ', text)  # Hapus simbol khusus
    text = clean_symbol.sub('', text)  # Hapus simbol yang tidak diperlukan
    text = stemmer.stem(text)  # Stemming
    text = ' '.join(word for word in text.split() if word not in stopwords)  # Hapus stopwords
    return text

# Pembersihan teks pada kolom 'judul_prosessing'
jurnal_df['desc_clean'] = jurnal_df['judul_prosessing'].apply(clean_text)

# Konversi teks ke matriks TF-IDF
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0)
tfidf_matrix = tfidf_vectorizer.fit_transform(jurnal_df['desc_clean'])

# Hitung cosine similarity
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Buat indeks untuk pencarian rekomendasi
indices = pd.Series(jurnal_df['judul_prosessing'])

# Fungsi rekomendasi
def recommendations(name, top=10):
    """
    Memberikan rekomendasi berdasarkan nama yang diberikan
    Args:
        name (str): Judul yang ingin dicari rekomendasinya.
        top (int): Jumlah rekomendasi yang diinginkan.
    Returns:
        list: Daftar rekomendasi dengan skor kemiripan.
    """
    if name not in indices.values:
        return f"Error: '{name}' tidak ditemukan dalam dataset. Pastikan nama sudah benar."

    # Ambil indeks dari nama yang diminta
    idx = indices[indices == name].index[0]

    # Ambil skor kesamaan
    similarity_scores = pd.Series(cosine_sim_matrix[idx]).sort_values(ascending=False)

    # Dapatkan rekomendasi teratas (lewati indeks pertama karena itu dirinya sendiri)
    top_indexes = list(similarity_scores.iloc[1:top+1].index)
    recommended_list = [
        f"{jurnal_df['judul_prosessing'].iloc[i]} - Similarity: {similarity_scores[i]:.2f}" 
        for i in top_indexes
    ]

    return recommended_list

# Contoh penggunaan
try:
    hasil_rekomendasi = recommendations('teknologi', top=5)
    print("\nRekomendasi:")
    for rekomendasi in hasil_rekomendasi:
        print(rekomendasi)
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
