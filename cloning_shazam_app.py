import os
import gdown
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import assemblyai as aai

# -------------------------------
# Load API Key from Streamlit Secrets
# -------------------------------
aai.settings.api_key = st.secrets["general"]["ASSEMBLYAI_API_KEY"]

# -------------------------------
# Download Database from Google Drive using gdown
# -------------------------------
# Google Drive File ID (make sure this file is publicly accessible)
GDRIVE_FILE_ID = "1bKx176TVlxQbMEFuDyzSBmceLapQYHT8"

def download_db_from_drive(file_id, output_path="eng_subtitles_database.db"):
    url = f"https://drive.google.com/uc?id={file_id}"
    st.write("Downloading database from:", url)
    gdown.download(url, output_path, quiet=False)
    return output_path

DB_PATH = "eng_subtitles_database.db"
if not os.path.exists(DB_PATH):
    with st.spinner("Downloading database from Google Drive..."):
        DB_PATH = download_db_from_drive(GDRIVE_FILE_ID)
        st.success("✅ Database downloaded successfully!")

# Debug info: file existence and size
st.write(f"File exists: {os.path.exists(DB_PATH)}")
st.write(f"File size: {os.path.getsize(DB_PATH)} bytes")

# -------------------------------
# Load Subtitle Data from the SQLite Database
# -------------------------------
def load_subtitles(db_path):
    try:
        conn = sqlite3.connect(db_path)
        # Assuming the table is named "zipfiles" with columns num, name, content
        df = pd.read_sql("SELECT num, name, content FROM zipfiles", conn)
        conn.close()
        # Convert binary content to text using latin-1 decoding
        df["decoded_text"] = df["content"].apply(lambda x: x.decode("latin-1") if isinstance(x, bytes) else "")
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading database: {e}")
        return None

df = load_subtitles(DB_PATH)
if df is None or df.empty:
    st.error("⚠️ Database could not be loaded. Please verify that the Google Drive file is a valid SQLite database and has the correct structure.")
    st.stop()
else:
    st.success(f"✅ Loaded {len(df)} subtitles successfully!")

# -------------------------------
# Create FAISS Index from Subtitle Embeddings
# -------------------------------
# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(df):
    embeddings = model.encode(df["decoded_text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings)
    d = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, df

index, df = create_faiss_index(df)

# -------------------------------
# Streamlit UI for Audio Transcription & Subtitle Search
# -------------------------------
st.set_page_config(page_title="Shazam Clone: Audio-to-Subtitle Search", layout="wide")
st.title("🎵 Shazam Clone: Audio-to-Subtitle Search")
st.markdown("Upload an **audio file**, transcribe it to text, and retrieve the most relevant subtitle segments!")

uploaded_audio = st.file_uploader("🎤 Upload an Audio File", type=["mp3", "wav", "m4a"])

if uploaded_audio:
    st.audio(uploaded_audio, format="audio/mp3")
    if st.button("🎙 Transcribe Audio"):
        with st.spinner("Transcribing..."):
            # Transcribe audio using AssemblyAI
            transcript = aai.Transcriber().transcribe(uploaded_audio)
            query_text = transcript.text
            st.success("✅ Transcription Complete!")
            st.text_area("🎧 Transcribed Text", query_text, height=150)

            # Search Subtitles Using FAISS
            def search_subtitles(query_text, index, df, top_k=5):
                query_embedding = model.encode([query_text])
                D, I = index.search(query_embedding, top_k)
                return df.iloc[I[0]]

            results = search_subtitles(query_text, index, df)
            st.markdown("## 🔍 **Top Matching Subtitles**")
            for _, row in results.iterrows():
                st.markdown(f"""
                **🎬 Movie:** {row["name"]}
                - **📜 Subtitle:** `{row["decoded_text"][:200]}...`
                - 🔗 **[View on OpenSubtitles](https://www.opensubtitles.org/en/subtitles/{row["num"]})**
                """)

st.sidebar.header("🔧 Settings")
st.sidebar.markdown("""
- **Powered by:** `FAISS + AssemblyAI + Sentence Transformers`
- **Database:** `SQLite (.db) file (downloaded from Google Drive)`
- **Search Mechanism:** `Semantic Search (FAISS)`
""")
st.sidebar.write("📌 Built with ❤️ using **Streamlit** 🚀")



