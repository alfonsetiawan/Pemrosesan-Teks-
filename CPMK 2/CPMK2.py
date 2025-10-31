import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import difflib

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("youtube_comments.csv")

# --- Kamus normalisasi kata tidak baku (perluasan)
normalisasi = {
    # general slang
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "bgt": "banget", "bngt": "banget", "bener": "benar", "yg": "yang",
    "dgn": "dengan", "sm": "sama", "aja": "saja", "tp": "tapi", "jd": "jadi",
    "krn": "karena", "dr": "dari", "trs": "terus", "si": "", "u": "kamu",
    # conversational fillers
    "nih": "", "dong": "", "deh": "", "loh": "", "lah": "",
    # common typos / slang mapped explicitly
    "ipong": "iphone", "iphon": "iphone", "iphonex": "iphone", "iphonr": "iphone",
    "samsng": "samsung", "samsun": "samsung", "xiaomii": "xiaomi",
    "opp0": "oppo", "vivoo": "vivo", "redmi1": "redmi",
    "gogle": "google", "googl": "google",
    "yt": "youtube", "ytb": "youtube",
    "ig": "instagram", "fb": "facebook",
    "klo": "kalau", "kpn": "kapan"
}

# --- Daftar kata baku / canonical untuk fuzzy matching
canonical_words = [
    "iphone", "samsung", "xiaomi", "oppo", "vivo", "redmi", "realme",
    "google", "youtube", "instagram", "facebook", "twitter", "tiktok",
    "laptop", "keyboard", "headphone", "earphone", "charger", "kamera",
    "lagu", "video", "film", "bagus", "jelek", "mantap", "kerennya",
    "keren", "cantik", "ganteng", "murah", "mahal", "original", "ori",
    "karena", "tapi", "jadi", "terus", "suka", "tidak", "banget"
]
# tambahkan canonical dari normalisasi dict values agar lebih komprehensif
canonical_words = list(set(canonical_words + list(normalisasi.values())))

# --- Fungsi koreksi fuzzy untuk token yang bukan stopword
def fuzzy_correction(token, cutoff=0.82):
    """
    Periksa token terhadap kamus normalisasi dulu. 
    Jika tidak ditemukan, gunakan difflib.get_close_matches untuk
    mencocokkan token ke canonical_words.
    """
    if not token:
        return token
    # kalau ada di kamus normalisasi, return mapping langsung
    if token in normalisasi:
        return normalisasi[token]
    # jangan koreksi kata sangat pendek (>=4 chars minimal)
    if len(token) < 4:
        return token
    # gunakan fuzzy match untuk menemukan kata baku serupa
    matches = difflib.get_close_matches(token, canonical_words, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    return token

# --- Fungsi preprocessing (menggunakan fuzzy_correction)
def clean_text(text):
    if pd.isna(text):
        return ""
    # lowercase
    text = text.lower()
    # hapus URL
    text = re.sub(r"http\S+|www\S+", "", text)
    # hapus mention & hashtag
    text = re.sub(r"@\w+|#\w+", "", text)
    # hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    # hapus angka
    text = re.sub(r"\d+", "", text)
    # hapus emoji / non-ascii
    text = text.encode('ascii', 'ignore').decode('ascii')
    # tokenisasi sederhana dan normalisasi + koreksi fuzzy
    tokens = text.split()
    # pertama: mapping eksplisit normalisasi (untuk menangani kata sangat umum)
    tokens = [normalisasi.get(t, t) for t in tokens]
    # selanjutnya: fuzzy correction terhadap canonical_words untuk token panjang
    tokens = [fuzzy_correction(t) for t in tokens]
    # hapus stopwords bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 0]
    return " ".join(tokens).strip()

# --- Terapkan preprocessing
df['cleaned'] = df['comment'].astype(str).apply(clean_text)

# --- Hapus duplikat & baris kosong
df = df.drop_duplicates(subset=['cleaned'])
df = df[df['cleaned'].str.len() > 0]

# --- Fungsi buat n-grams (bigram/trigram)
def make_ngrams(text, n):
    tokens = word_tokenize(text)
    if len(tokens) < n:
        return []
    return [" ".join(g) for g in ngrams(tokens, n)]

df['bigram'] = df['cleaned'].apply(lambda x: make_ngrams(x, 2))
df['trigram'] = df['cleaned'].apply(lambda x: make_ngrams(x, 3))

# --- WordCloud 
all_words = " ".join(df['cleaned'])
if len(all_words.strip()) > 0:
    wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(all_words)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud Komentar YouTube", fontsize=16)
    plt.show()

# --- Simpan hasil akhir
df[['cleaned', 'bigram', 'trigram']].to_csv("youtube_comments_processed.csv", index=False, encoding='utf-8-sig')
print("✅ File preprocessing selesai → youtube_comments_processed.csv")
