
# Anggota kelompok
# latif ardiansyah 22.12.2599
# reyhan dwi wira allofadieka 22.12.2563

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Cek apakah file ada
try:
    jurnal_df = pd.read_excel('jurnal_sinta.xlsx')
except FileNotFoundError:
    raise FileNotFoundError("File 'jurnal_sinta.xlsx' tidak ditemukan. Pastikan file ada di direktori yang benar.")

# Hapus data kosong
jurnal_df = jurnal_df[jurnal_df['judul_prosessing'].notnull()]

# Inisialisasi pembersihan teks
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
stopwords = sastrawi.get_stop_words()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi pembersihan teks
def clean_text(text):
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = stemmer.stem(text)
    text = ' '.join(word for word in text.split() if word not in stopwords)
    return text

# Pembersihan teks pada kolom 'judul_prosessing'
jurnal_df['desc_clean'] = jurnal_df['judul_prosessing'].apply(clean_text)

# Konversi teks ke matriks TF-IDF
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0)
tfidf_matrix = tf.fit_transform(jurnal_df['desc_clean'])

# Hitung cosine similarity
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Buat indeks untuk pencarian rekomendasi
indices = pd.Series(jurnal_df.index)

# Fungsi rekomendasi
def recommendations(name, top=10):
    recommended_list = []

    if name not in indices.values:
        return f"Error: '{name}' not found in the indices. Available indices are: {list(indices.values)}"

    # Ambil indeks dari nama yang diminta
    idx = indices[indices == name].index[0]

    # Ambil skor kesamaan
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)

    # Dapatkan rekomendasi teratas
    top_indexes = list(score_series.iloc[1:top+1].index)  # Skip indeks pertama (item itu sendiri)

    for i in top_indexes:
        recommended_list.append(f"{list(jurnal_df.index)[i]} - Similarity: {score_series[i]:.2f}")

    return recommended_list

# Contoh penggunaan
print(recommendations('teknologi', top=5))
