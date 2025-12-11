import pandas as pd

# 1. LOAD DATA
df = pd.read_csv("youtube_comments.csv")

# 2. DEFINISI KATA KUNCI
positive_words = [
    "bagus", "mantap", "keren", "suka", "recommended", "puas", "wow",
    "terbaik", "mantul", "baguss", "bagus banget"
]

negative_words = [
    "jelek", "buruk", "kecewa", "marah", "ga suka", "gak suka",
    "sampah", "parah", "benci", "error", "lag", "nge lag"
]

def is_positive(text):
    text = str(text).lower()
    return any(w in text for w in positive_words)

def is_negative(text):
    text = str(text).lower()
    return any(w in text for w in negative_words)

# 3. FILTER KOMENTAR
df_positif = df[df["comment"].apply(is_positive)]
df_negatif = df[df["comment"].apply(is_negative)]
df_netral  = df[~df["comment"].apply(is_positive) & ~df["comment"].apply(is_negative)]

# JUMLAH TERSEDIA
jumlah_pos = len(df_positif)
jumlah_neg = len(df_negatif)
jumlah_neu = len(df_netral)

print("Jumlah positif:", jumlah_pos)
print("Jumlah negatif:", jumlah_neg)
print("Jumlah netral :", jumlah_neu)

# 4. AMBIL SAMPEL SESUAI JUMLAH
data_positif = df_positif.sample(min(100, jumlah_pos), random_state=42)
data_negatif = df_negatif.sample(min(100, jumlah_neg), random_state=42)
data_netral  = df_netral.sample(min(50, jumlah_neu), random_state=42)

# 5. TAMBAH LABEL
data_positif["label"] = "positif"
data_negatif["label"] = "negatif"
data_netral["label"]  = "netral"

# 6. GABUNG DATASET FINAL
dataset_final = pd.concat([data_positif, data_negatif, data_netral], ignore_index=True)

dataset_final.to_csv("dataset_sentimen_final.csv", index=False)

print("\nDataset final selesai dibuat!")
print("Positif:", len(data_positif))
print("Negatif:", len(data_negatif))
print("Netral :", len(data_netral))

# 7. BUAT DATA SISA
index_final = dataset_final.index
data_sisa = df.drop(dataset_final.index, errors='ignore')

data_sisa.to_csv("data_sisa.csv", index=False)

print("\nData sisa berhasil dibuat!")
print("Jumlah data sisa:", len(data_sisa))
