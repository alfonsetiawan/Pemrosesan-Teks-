from googleapiclient.discovery import build
import pandas as pd
from tqdm import tqdm

# Ganti dengan API key kamu
API_KEY = "AIzaSyAo6DnpYatVlKPCO3hccINeclkL5UoNwcs"
VIDEO_ID = "lFR4utbwliw"  # contoh: 'dQw4w9WgXcQ'
youtube = build('youtube', 'v3', developerKey=API_KEY)

comments = []
next_page_token = None

print("Mengambil komentar dari video:", VIDEO_ID)

while True:
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=VIDEO_ID,
        maxResults=100,
        pageToken=next_page_token,
        textFormat="plainText"
    )
    response = request.execute()

    for item in response['items']:
        snippet = item['snippet']['topLevelComment']['snippet']
        comments.append({
            "author": snippet['authorDisplayName'],
            "comment": snippet['textDisplay'],
            "likeCount": snippet['likeCount'],
            "publishedAt": snippet['publishedAt']
        })

    next_page_token = response.get('nextPageToken')

    if not next_page_token or len(comments) >= 1000:
        break

print(f"Total komentar dikumpulkan: {len(comments)}")

# Simpan ke CSV
df = pd.DataFrame(comments)
df.to_csv("youtube_comments.csv", index=False, encoding='utf-8-sig')
print("Komentar berhasil disimpan ke youtube_comments.csv")
