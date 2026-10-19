# 🎵 Emotion-Aware Music Player

Your face picks the playlist. Take a photo, the AI detects your emotion, and instantly queues music that matches your mood — no aux cord drama.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![FER](https://img.shields.io/badge/FER-22.5.1-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=flat-square&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

🌐 **Live Demo:** [Try it here](https://emotion-music-player-foj8.onrender.com/)

> ⚠️ First load may take 30-50 seconds as the server wakes from idle (Render free tier).

---

## 🎯 What It Does

- 📷 **Captures your photo** via webcam directly in the browser
- 🧠 **Detects your emotion** — happy, sad, angry, neutral, surprised, fear, disgust
- 🎵 **Queues matching music** — curated YouTube playlists per emotion
- 📊 **Shows full emotion breakdown** — confidence scores for all 7 emotions
- 🎬 **Embedded player** — plays music right in the app

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| FER (Facial Expression Recognition) | Emotion detection from face |
| OpenCV | Image processing |
| TensorFlow CPU | ML backend for FER |
| Streamlit | Web app + camera input |
| Docker | Containerized deployment on Render |

---

## 🧠 Emotion → Mood Mapping

| Detected Emotion | Music Mood | Example Artists |
|-----------------|------------|----------------|
| 😄 Happy | Feel-good bangers | Pharrell, Bruno Mars, Lizzo |
| 😢 Sad | Emotional ballads | Adele, Coldplay, Bon Iver |
| 😤 Angry | High-energy rock | Linkin Park, RATM, Imagine Dragons |
| 😐 Neutral | Chill lofi/indie | Tame Impala, MGMT, Lofi Radio |
| 😲 Surprised | Pop hits | The Weeknd, Dua Lipa, Harry Styles |
| 😨 Anxious/Fear | Calming classical | Einaudi, Debussy, Satie |
| 😒 Disgusted | Reset vibes | SZA, Olivia Rodrigo |

---

## 🚀 Setup & Run Locally

```bash
git clone https://github.com/Navld14/emotion-music-player.git
cd emotion-music-player
pip install -r requirements.txt
streamlit run app.py
```

### Or run with Docker
```bash
docker build -t emotion-music .
docker run -p 8501:8501 emotion-music
```

---

## 📁 Project Structure

```
emotion-music-player/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker config for Render
└── README.md
```

---

## 🙋 Author

**Naveen Ladha**
[GitHub](https://github.com/Navld14)

---

## 📄 License

MIT License — free to use and modify.
