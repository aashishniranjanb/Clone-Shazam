import os
import json
import numpy as np
import streamlit as st
import faiss
import assemblyai as aai
import sqlite3
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

# Load AssemblyAI API Key from Streamlit Secrets
api_key = st.secrets["general"]["ASSEMBLYAI_API_KEY"]  
aai.settings.api_key = api_key

st.write("✅ API Key Loaded Successfully!")

# Google Drive File ID
GDRIVE_FILE_ID = "1bKx176TVlxQbMEFuDyzSBmceLapQYHT8"  

# Function to Download DB from Google Drive
def download_db_from_drive(file_id, output_path="eng_subtitles_database.db"):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    return output_path

# Check if DB file exists, if not, download it
DB_PATH = "eng_subtitles_database.db"
if not os.path.exists(DB_PATH):
    with st.spinner("Downloading database from Google Drive..."):
        DB_PATH = download_db_from_drive(GDRIVE_FILE_ID)
        st.success("✅ Database downloaded successfully!")

# Load Subtitle Data
def load_subtitles(db_path):
    if not os.path.exists(db_path):
        st.error("⚠️ Database file not found! Please upload the .db file.")
        return None  # Stop execution

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT num, name, content FROM zipfiles", conn)
    except Exception as e:
        st.error(f"⚠️ Error loading database: {e}")
        df = None  # Return None on failure
    finally:
        conn.close()

    if df is not None:
        # Convert binary content to text
        df["decoded_text"] = df["content"].apply(lambda x: x.decode("latin-1") if isinstance(x, bytes) else "")

    return df


df = load_subtitles(DB_PATH)

# Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create FAISS Index
def create_faiss_index(df):
    embeddings = model.encode(df["decoded_text"].tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, df

index, df = create_faiss_index(df)

# Streamlit UI
st.set_page_config(page_title="Shazam Clone: Audio-to-Subtitle Search", layout="wide")
st.title("🎵 Shazam Clone: Audio-to-Subtitle Search")
st.markdown("Upload an **audio file**, transcribe it to text, and retrieve the most relevant subtitle segments!")

# 🔹 File Upload Section
uploaded_audio = st.file_uploader("🎤 Upload an Audio File", type=["mp3", "wav", "m4a"])

if uploaded_audio:
    st.audio(uploaded_audio, format="audio/mp3")

    if st.button("🎙 Transcribe Audio"):
        with st.spinner("Transcribing..."):
            query_text = aai.Transcriber().transcribe(uploaded_audio).text
            st.success("✅ Transcription Complete!")
            st.text_area("🎧 Transcribed Text", query_text, height=150)

            # Search Subtitles
            def search_subtitles(query_text, index, df, top_k=5):
                query_embedding = model.encode([query_text])
                D, I = index.search(query_embedding, top_k)
                return df.iloc[I[0]]

            results = search_subtitles(query_text, index, df)

            # Display Results
            st.markdown("## 🔍 **Top Matching Subtitles**")
            for _, row in results.iterrows():
                st.markdown(f"""
                **🎬 Movie:** {row["name"]}
                - **📜 Subtitle:** `{row["decoded_text"]}`
                - 🔗 **[View on OpenSubtitles](https://www.opensubtitles.org/en/subtitles/{row["num"]})**
                """)

st.sidebar.header("🔧 Settings")
st.markdown("---")
st.markdown("**Developed by [Aashish Niranjan BarathyKannan](https://www.linkedin.com/in/aashishniranjanb/)** | [GitHub](https://github.com/aashishniranjanb)")
st.sidebar.markdown("""
- **Powered by:** `FAISS + AssemblyAI + Sentence Transformers`
- **Database:** `SQLite (.db) file (from Google Drive)`
- **Search Mechanism:** `Semantic Search (FAISS)`
""")
st.sidebar.write("📌 Built with ❤️ using **Streamlit** 🚀")


