import streamlit as st
import cv2
import numpy as np
from fer import FER
from PIL import Image
import random

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Music Player 🎵",
    page_icon="🎵",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero { text-align: center; padding: 1.5rem 0 0.5rem 0; }
.hero h1 {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #ec4899, #8b5cf6, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero p { color: #888; font-size: 1rem; margin-top: -6px; }

.emotion-card { background: #111; border: 1px solid #222; border-radius: 16px; padding: 24px; text-align: center; }
.emotion-emoji { font-size: 4rem; line-height: 1; }
.emotion-label { font-size: 1.4rem; font-weight: 700; margin-top: 8px; }

.now-playing { background: linear-gradient(135deg, #0f0f1a, #1a0f2e); border: 1px solid #2d1b69; border-radius: 16px; padding: 20px; margin-top: 16px; }
.track-title { font-size: 1.1rem; font-weight: 600; color: #e2d9f3; }
.track-artist { font-size: 0.85rem; color: #7c5cbf; margin-top: 4px; }

.playlist-item { background: #111; border: 1px solid #1e1e1e; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
.section-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #555; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Mood → Music mapping ───────────────────────────────────────────────────────
MOOD_DATA = {
    "happy": {
        "emoji": "😄", "color": "#facc15", "label": "Happy",
        "description": "You're radiating good vibes! Here's some feel-good music 🎉",
        "tracks": [
            ("Happy - Pharrell Williams", "ZbZSe6N_BXs"),
            ("Can't Stop The Feeling - Justin Timberlake", "ru0K8uYEZWw"),
            ("Good as Hell - Lizzo", "SmbmeOgWsqE"),
            ("Uptown Funk - Bruno Mars", "OPf0YbXqDm0"),
            ("Shake It Off - Taylor Swift", "nfWlot6h_JM"),
        ]
    },
    "sad": {
        "emoji": "😢", "color": "#60a5fa", "label": "Sad",
        "description": "Feeling blue? Let the music heal 💙",
        "tracks": [
            ("Someone Like You - Adele", "hLQl3WQQoQ0"),
            ("Fix You - Coldplay", "k4V3Mo61fJM"),
            ("The Night We Met - Lord Huron", "KtlgYxa6BMU"),
            ("Let Her Go - Passenger", "RBumgq5yVrA"),
            ("Skinny Love - Bon Iver", "ssdgB-ia6v4"),
        ]
    },
    "angry": {
        "emoji": "😤", "color": "#f87171", "label": "Angry",
        "description": "Let it out! Here's some music to match the energy 🔥",
        "tracks": [
            ("Break Stuff - Limp Bizkit", "ZpUYjpKg9KY"),
            ("Killing In The Name - RATM", "bWXazVeVsGs"),
            ("Given Up - Linkin Park", "0xyxtzD54rM"),
            ("Believer - Imagine Dragons", "7wtfhZwyrcc"),
            ("In The End - Linkin Park", "eVTXPUF4Oz4"),
        ]
    },
    "neutral": {
        "emoji": "😐", "color": "#94a3b8", "label": "Neutral",
        "description": "Calm and composed. Here's some chill music 🎧",
        "tracks": [
            ("Lofi Hip Hop Radio", "jfKfPfyJRdk"),
            ("Chill Vibes - Study Music", "5qap5aO4i9A"),
            ("The Less I Know The Better - Tame Impala", "2SUwOgmvzK4"),
            ("Redbone - Childish Gambino", "Kp7eSUU9oy8"),
            ("Electric Feel - MGMT", "MmZexg8sxyk"),
        ]
    },
    "surprise": {
        "emoji": "😲", "color": "#fb923c", "label": "Surprised",
        "description": "Wide-eyed and wondering! Here's something unexpected 🎊",
        "tracks": [
            ("Blinding Lights - The Weeknd", "4NRXx6U8ABQ"),
            ("Levitating - Dua Lipa", "TUVcZfQe-Kw"),
            ("As It Was - Harry Styles", "H5v3kku4y6Q"),
            ("Peaches - Justin Bieber", "hM8UXISVqSI"),
            ("STAY - The Kid LAROI", "kTJczUoc26U"),
        ]
    },
    "fear": {
        "emoji": "😨", "color": "#a78bfa", "label": "Anxious",
        "description": "Take a breath. Here's some calming music 🌊",
        "tracks": [
            ("Weightless - Marconi Union", "UfcAVejslrU"),
            ("Claire de Lune - Debussy", "WNcsUNKnbDY"),
            ("Gymnopédie No. 1 - Satie", "S-Xm7s9eGxU"),
            ("Experience - Ludovico Einaudi", "hN_q-_nGv4U"),
            ("Divenire - Ludovico Einaudi", "7-oHBBfkFwM"),
        ]
    },
    "disgust": {
        "emoji": "😒", "color": "#4ade80", "label": "Disgusted",
        "description": "Not feeling it? Here's music to reset your vibe 🔄",
        "tracks": [
            ("Good Days - SZA", "FO-b8R4HCAM"),
            ("drivers license - Olivia Rodrigo", "ZmDBbnmKpqQ"),
            ("Mood - 24kGoldn", "L00pDRklPgk"),
            ("Brutal - Olivia Rodrigo", "AKlD-ItGBD8"),
            ("traitor - Olivia Rodrigo", "OuFEBMvNaOA"),
        ]
    }
}

# ── Load FER detector ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading emotion detector...")
def load_detector():
    return FER(mtcnn=False)


# ── Detect emotion ─────────────────────────────────────────────────────────────
def detect_emotion(frame, detector):
    try:
        result = detector.detect_emotions(frame)
        if not result:
            return "neutral", 0.0, {}
        emotions = result[0]["emotions"]
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant] * 100
        return dominant, round(confidence, 1), emotions
    except Exception:
        return "neutral", 0.0, {}


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🎵 Emotion Music Player</h1>
    <p>Your face picks the playlist. No aux cord drama.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

detector = load_detector()

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown("<div class='section-label'>📷 Camera Feed</div>", unsafe_allow_html=True)
    camera_input = st.camera_input("", label_visibility="collapsed")

with right:
    st.markdown("<div class='section-label'>🧠 Detected Emotion</div>", unsafe_allow_html=True)

    if camera_input is None:
        st.markdown("""
        <div class='emotion-card'>
            <div class='emotion-emoji'>🎭</div>
            <div class='emotion-label' style='color:#444'>Waiting for camera...</div>
            <div style='color:#666; font-size:0.85rem; margin-top:8px'>Take a photo to detect your emotion</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        img = Image.open(camera_input).convert("RGB")
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with st.spinner("Reading your face... 🧠"):
            emotion, confidence, all_emotions = detect_emotion(frame_bgr, detector)

        mood = MOOD_DATA.get(emotion, MOOD_DATA["neutral"])

        st.markdown(f"""
        <div class='emotion-card'>
            <div class='emotion-emoji'>{mood['emoji']}</div>
            <div class='emotion-label' style='color:{mood['color']}'>{mood['label']}</div>
            <div style='color:#666; font-size:0.85rem; margin-top:4px'>Confidence: {confidence:.1f}%</div>
            <div style='color:#666; font-size:0.85rem; margin-top:12px'>{mood['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        if all_emotions:
            with st.expander("📊 Full emotion breakdown"):
                sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
                for emo, score in sorted_emotions:
                    emo_data = MOOD_DATA.get(emo, {})
                    color = emo_data.get("color", "#888")
                    pct = score * 100
                    st.markdown(f"""
                    <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px'>
                        <div style='width:90px; color:#888; font-size:0.8rem; text-transform:capitalize'>{emo}</div>
                        <div style='flex:1; background:#1a1a1a; border-radius:999px; height:8px'>
                            <div style='width:{min(pct,100):.0f}%; background:{color}; height:8px; border-radius:999px'></div>
                        </div>
                        <div style='width:45px; text-align:right; font-size:0.8rem; color:{color}'>{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Music
        st.markdown("<div class='section-label' style='margin-top:24px'>🎵 Your Playlist</div>", unsafe_allow_html=True)
        tracks = mood["tracks"]
        selected_track = random.choice(tracks)
        track_name, video_id = selected_track

        st.markdown(f"""
        <div class='now-playing'>
            <div style='font-size:0.7rem; color:#7c5cbf; letter-spacing:2px; text-transform:uppercase; margin-bottom:8px'>▶ Now Playing</div>
            <div class='track-title'>{track_name}</div>
            <div class='track-artist'>via YouTube</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top:16px; border-radius:12px; overflow:hidden'>
            <iframe width="100%" height="200"
                src="https://www.youtube.com/embed/{video_id}?autoplay=0"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen style="border-radius:12px">
            </iframe>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-label' style='margin-top:20px'>📋 More tracks</div>", unsafe_allow_html=True)
        for name, vid_id in tracks:
            if vid_id != video_id:
                st.markdown(f"""
                <a href='https://www.youtube.com/watch?v={vid_id}' target='_blank' style='text-decoration:none'>
                    <div class='playlist-item'>
                        <span style='color:#ccc; font-size:0.9rem'>🎵 {name}</span>
                        <span style='color:#555; font-size:0.75rem; float:right'>▶ Play</span>
                    </div>
                </a>
                """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#333; font-size:0.8rem'>Built with Python · FER · OpenCV · Streamlit</p>", unsafe_allow_html=True)
