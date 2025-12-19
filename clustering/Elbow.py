# ======================================
# PREPROCESSING + ELBOW + CLUSTERING
# ======================================

import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --------------------------------------
# Download stopword (cukup sekali)
# --------------------------------------
nltk.download('stopwords')

# --------------------------------------
# Load Dataset
# --------------------------------------
df = pd.read_csv('youtube_comments.csv')
df = df.dropna()

# Batasi data agar proses cepat
df = df.head(1000)

teks = df['comment'].astype(str)

# --------------------------------------
# Kamus Normalisasi Typo / Slang
# --------------------------------------
normalisasi_kata = {
    'gk': 'tidak',
    'ga': 'tidak',
    'nggak': 'tidak',
    'tdk': 'tidak',
    'bgt': 'banget',
    'yg': 'yang',
    'aja': 'saja',
    'dgn': 'dengan',
    'krn': 'karena',
    'sm': 'sama',
    'pdhl': 'padahal'
}

# --------------------------------------
# Stopword & Stemmer
# --------------------------------------
stopword_indo = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

# --------------------------------------
# Fungsi Preprocessing
# --------------------------------------
def preprocess_teks(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [normalisasi_kata.get(kata, kata) for kata in tokens]
    tokens = [kata for kata in tokens if kata not in stopword_indo]
    tokens = [stemmer.stem(kata) for kata in tokens]
    return ' '.join(tokens)

# --------------------------------------
# Terapkan Preprocessing
# --------------------------------------
df['comment_clean'] = teks.apply(preprocess_teks)

# --------------------------------------
# TF-IDF
# --------------------------------------
vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=5,
    max_df=0.9
)

X = vectorizer.fit_transform(df['comment_clean'])

# --------------------------------------
# ELBOW METHOD
# --------------------------------------
inertia = []
K = range(2, 8)

for k in K:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=5
    )
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow
plt.figure()
plt.plot(list(K), inertia, marker='o')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# --------------------------------------
# CLUSTERING FINAL
# --------------------------------------
# Tentukan k optimal berdasarkan grafik elbow
k_optimal = 3   # GANTI sesuai hasil elbow

kmeans_final = KMeans(
    n_clusters=k_optimal,
    random_state=42,
    n_init=10
)

df['cluster'] = kmeans_final.fit_predict(X)

# --------------------------------------
# OUTPUT HASIL
# --------------------------------------
print("Jumlah data per cluster:")
print(df['cluster'].value_counts())

print("\nContoh hasil clustering:")
print(df[['comment', 'comment_clean', 'cluster']].head(10))
