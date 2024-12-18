import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st

# Preprocessing setup
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
factory = StopWordRemoverFactory()
stopworda = factory.get_stop_words()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Text cleaning function
def clean_text(text):
    text = text.lower()  
    text = clean_spcl.sub(' ', text)  
    text = clean_symbol.sub('', text)  
    text = ' '.join(word for word in text.split() if word not in stopworda)  # Hapus stopwords
    text = stemmer.stem(text)  # Stemming
    return text

# Load dataset
jurnal_df = pd.read_excel('jurnal_sinta_tek.xlsx')

# Process text data
jurnal_df['judul_prosessing'] = jurnal_df['judul_prosessing'].apply(clean_text)
jurnal_df.reset_index(inplace=True)
jurnal_df.set_index('judul_prosessing', inplace=True)

# Initialize TF-IDF Vectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1)
tfidf_matrix = tf.fit_transform(jurnal_df.index)
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(jurnal_df.index)

# Recommendation function
def recommendations(name, top=10):
    recommended_jurnal = []

    cleaned_name = clean_text(name)
    input_tokens = set(cleaned_name.split())
    filtered_indices = []
    
    # Filter matching journals
    for idx in jurnal_df.index:
        if any(token in clean_text(idx) for token in input_tokens):
            filtered_indices.append(idx)

    if not filtered_indices:  # If no matching journals
        return [f"Tidak ada jurnal yang cocok dengan kata kunci '{name}'"]

    filtered_df = jurnal_df.loc[filtered_indices]
    filtered_numeric_indices = [list(jurnal_df.index).index(i) for i in filtered_indices]

    # Recompute TF-IDF for filtered data
    tfidf_matrix_filtered = tfidf_matrix[filtered_numeric_indices]
    cos_sim_filtered = cosine_similarity(tfidf_matrix_filtered, tfidf_matrix_filtered)

    # Sort the scores
    score_series = pd.Series(cos_sim_filtered[0]).sort_values(ascending=False)

    # Ambil top indeks setelah jurnal pertama
    top_indexes = list(score_series.iloc[1:top+1].index)

    # If input journal is found in filtered list, add it to the front
    input_position = next((i for i, idx in enumerate(filtered_indices) if clean_text(idx) == cleaned_name), None)
    if input_position is not None:
        top_indexes.insert(0, input_position)  # Insert input journal in front if it exists

    # Final recommended journals based on sorted indexes
    recommended_jurnal = [filtered_indices[i] for i in top_indexes]

    return recommended_jurnal[:top]

st.title("Sistem Rekomendasi jurnal Sinta Teknologi")

place_input = st.text_input("Masukkan nama jurnal favorit Anda:")

num_recommendations = st.slider("Pilih jumlah rekomendasi jurnal", min_value=1, max_value=59, value=5)

if st.button("Cari Recomendasi"):
    if place_input:
        with st.spinner("Mencari rekomendasi..."):
            hasil_rekomendasi = recommendations(place_input, top=num_recommendations)
            st.write("Rekomendasi jurnal untuk Anda:")
            for idx, jurnal in enumerate(hasil_rekomendasi, start=1):
                st.write(f"{idx}. {jurnal}")


st.dataframe(jurnal_df)
