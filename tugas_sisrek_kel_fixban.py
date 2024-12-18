import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st

clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
factory = StopWordRemoverFactory()
stopworda = factory.get_stop_words()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()  
    text = clean_spcl.sub(' ', text)  
    text = clean_symbol.sub('', text)  
    text = ' '.join(word for word in text.split() if word not in stopworda)  # Hapus stopwords
    text = stemmer.stem(text)  # Stemming
    return text

jurnal_df = pd.read_excel('jurnal_sinta_tek.xlsx')

jurnal_df['judul_prosessing'] = jurnal_df['judul_prosessing'].apply(clean_text)
jurnal_df.reset_index(inplace=True)

jurnal_df.set_index('judul_prosessing', inplace=True)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1)

tfidf_matrix = tf.fit_transform(jurnal_df.index)

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(jurnal_df.index)
#
def recommendations(name, top=10):
    recommended_jurnal = []

    cleaned_name = clean_text(name)
    input_tokens = set(cleaned_name.split())
    filtered_indices = []
    for idx in jurnal_df.index:
        if any(token in clean_text(idx) for token in input_tokens):
            filtered_indices.append(idx)

    if not filtered_indices:  # Jika tidak ada hasil yang cocok
        return [f"Tidak ada jurnal yang cocok dengan kata kunci '{name}'"]

    filtered_df = jurnal_df.loc[filtered_indices]
    filtered_numeric_indices = [list(jurnal_df.index).index(i) for i in filtered_indices]

    tfidf_matrix_filtered = tfidf_matrix[filtered_numeric_indices]
    cos_sim_filtered = cosine_similarity(tfidf_matrix_filtered, tfidf_matrix_filtered)

    idx = filtered_numeric_indices[0] 

    score_series = pd.Series(cos_sim_filtered[0]).sort_values(ascending=False)

    top_indexes = list(score_series.iloc[1:top+1].index)

    input_position = next((i for i, idx in enumerate(filtered_indices) if clean_text(idx) == cleaned_name), None)
    
    if input_position is not None:
        top_indexes.insert(0, input_position)  

    recommended_jurnal = [filtered_indices[i] for i in top_indexes]

    return recommended_jurnal[:top]

st.title("Sistem Rekomendasi jurnal")

place_input = st.text_input("Masukkan nama jurnal favorit Anda:")
# Tampilkan tombol "Cari Recomendasi"
if st.button("Cari Recomendasi"):
    if place_input:
        with st.spinner("Mencari rekomendasi..."):
    
            hasil_rekomendasi = recommendations(place_input, top=5)
            st.write("Rekomendasi jurnal untuk Anda:")
            for jurnal in hasil_rekomendasi:
                st.write(jurnal)

jurnal_df = pd.read_excel('jurnal_sinta_tek.xlsx')
st.dataframe(jurnal_df)
