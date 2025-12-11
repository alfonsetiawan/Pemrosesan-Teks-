import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# ======================================================
# 1. LOAD DATA LATIH (DATA BERLABEL)
# ======================================================
df_train = pd.read_csv("dataset_sentimen_final.csv")

X_train = df_train["comment"]
y_train = df_train["label"]

# ======================================================
# 2. TRAIN TF-IDF
# ======================================================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# ======================================================
# 3. LATIH MODEL
# ======================================================
model = LogisticRegression(max_iter=3000)
model.fit(X_train_tfidf, y_train)

print("Model berhasil dilatih!")

# ======================================================
# 4. LOAD DATA SISA (TANPA LABEL)
# ======================================================
df_sisa = pd.read_csv("data_sisa.csv")

# pastikan kolomnya bernama "comment"
X_sisa = df_sisa["comment"]

# ======================================================
# 5. TRANSFORM + PREDIKSI LABEL
# ======================================================
X_sisa_tfidf = vectorizer.transform(X_sisa)
df_sisa["predicted_label"] = model.predict(X_sisa_tfidf)

# ======================================================
# 6. SIMPAN HASIL
# ======================================================
df_sisa.to_csv("data_sisa_berlabel.csv", index=False)

print("Selesai! File tersimpan sebagai data_sisa_berlabel.csv")
