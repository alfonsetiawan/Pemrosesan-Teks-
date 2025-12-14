import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================================
# 1. LOAD DATA
# ======================================================
df = pd.read_csv("data_sisa_berlabel.csv")

X = df["comment"]
y = df["predicted_label"]   # ganti ke "label" jika nama kolomnya label

# ======================================================
# 2. TF-IDF
# ======================================================
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# ======================================================
# 3. SPLIT DATA
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# ======================================================
# 4. TRAIN MODEL
# ======================================================
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# ======================================================
# 5. PREDIKSI
# ======================================================
y_pred = model.predict(X_test)

# ======================================================
# 6. EVALUASI
# ======================================================
print("AKURASI:")
print(accuracy_score(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))
