# Anggota kelompok
# Latif Ardiansyah 22.12.2599
# Reyhan Dwi Wira Allofadieka 22.12.2563

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Title aplikasi
st.title("Sistem Rekomendasi Jurnal SINTA")
st.write("Aplikasi ini memberikan rekomendasi berdasarkan judul yang dimasukkan.")

# Cek apakah file ada dan kolom 'judul_prosessing' tersedia
try:
    jurnal_df = pd.read_excel('jurnal_sinta.xlsx')
    if 'judul_prosessing' not in jurnal_df.columns:
        st.error("Kolom 'judul_prosessing' tidak ditemukan di file 'jurnal_sinta.xlsx'.")
        st.stop()
except FileNotFoundError:
    st.error("File 'jurnal_sinta.xlsx' tidak ditemukan. Pastikan file ada di direktori yang benar.")
    st.stop()
except Exception as e:
    st.error(f"Error saat membaca file: {e}")
    st.stop()

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
        return None

    # Ambil indeks dari nama yang diminta
    idx = indices[indices == name].index[0]

    # Ambil skor kesamaan
    similarity_scores = pd.Series(cosine_sim_matrix[idx]).sort_values(ascending=False)

    # Dapatkan rekomendasi teratas (lewati indeks pertama karena itu dirinya sendiri)
    top_indexes = list(similarity_scores.iloc[1:top+1].index)
    recommended_list = [
        (jurnal_df['judul_prosessing'].iloc[i], similarity_scores[i]) 
        for i in top_indexes
    ]

    return recommended_list

# Input pengguna
judul_input = st.text_input("Masukkan judul jurnal untuk rekomendasi:")

# Slider untuk menentukan jumlah rekomendasi
top_n = st.slider("Jumlah rekomendasi yang diinginkan:", 1, 20, 5)

# Tombol untuk memulai rekomendasi
if st.button("Cari Rekomendasi"):
    if judul_input:
        hasil_rekomendasi = recommendations(judul_input, top=top_n)
        if hasil_rekomendasi:
            st.subheader(f"Rekomendasi untuk: {judul_input}")
            for judul, similarity in hasil_rekomendasi:
                st.write(f"**{judul}** - Similarity: {similarity:.2f}")
        else:
            st.error(f"Judul '{judul_input}' tidak ditemukan dalam dataset.")
    else:
        st.error("Harap masukkan judul jurnal terlebih dahulu.")
