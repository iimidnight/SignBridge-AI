import av
import streamlit as st
import cv2
import joblib
import numpy as np
import mediapipe as mp
from pathlib import Path
from queue import Queue, Empty
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh
MODEL_PATH = Path("models/isl_sign_model.pkl")

@st.cache_resource
def load_sign_model():
    if not MODEL_PATH.exists():
        return None
    saved = joblib.load(MODEL_PATH)
    return saved["model"]

model = load_sign_model()
mp_hands = mp.solutions.hands

# Keep the prediction queue alive between Streamlit reruns.
if "prediction_queue" not in st.session_state:
    st.session_state.prediction_queue = Queue()

prediction_queue = st.session_state.prediction_queue


class SignProcessor(VideoProcessorBase):
    def __init__(self, result_queue):
        self.result_queue = result_queue
        self.hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
        self.last_label = None
        self.stable_count = 0
        self.no_hand_count = 0
        self.last_emitted_label = None

    def recv(self, video_frame):
        frame = video_frame.to_ndarray(format="bgr24")

        frame = cv2.resize(
        frame,
    (640, 480),
    interpolation=cv2.INTER_AREA
)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            self.no_hand_count = 0
            hand_data = []

            for hand in results.multi_hand_landmarks:
                xs = [point.x for point in hand.landmark]
                hand_data.append((sum(xs) / len(xs), hand))

            hand_data.sort(key=lambda item: item[0])
            features = []

            for i in range(2):
                if i < len(hand_data):
                    for point in hand_data[i][1].landmark:
                        features.extend([point.x, point.y, point.z])
                else:
                    features.extend([0.0] * 63)

            prediction = model.predict(
                np.array(features).reshape(1, -1)
            )[0]

            if prediction == self.last_label:
                self.stable_count += 1
            else:
                self.last_label = prediction
                self.stable_count = 1

            # Add a sign only after it remains stable for several frames.
            if (
                self.stable_count >= 10
                and prediction != self.last_emitted_label
            ):
                self.result_queue.put(prediction)
                self.last_emitted_label = prediction

        else:
            self.no_hand_count += 1
            self.stable_count = 0
            self.last_label = None

            # Releasing both hands allows the same sign to be added again.
            if self.no_hand_count >= 5:
                self.last_emitted_label = None

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


st.set_page_config(
    page_title="SignBridge AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
    background: linear-gradient(135deg, #090909 0%, #151515 55%, #090909 100%);
    color: #edf3ff;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
[data-testid="stSidebar"] {
    background: #101010;
    border-right: 1px solid rgba(148,163,184,.18);
}
.hero {
    padding: 1.7rem 1.8rem;
    border: 1px solid rgba(125,211,252,.25);
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(55,55,55,.45), rgba(28,28,28,.65));
    margin-bottom: 1.2rem;
}
.hero-kicker { color: #7dd3fc; font-size: .82rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
.hero h1 { color: #f8fbff; font-size: 2.5rem; margin: .35rem 0 .45rem 0; }
.hero p { color: #c8d7ec; font-size: 1.05rem; margin: 0; max-width: 760px; }
.panel {
    background: rgba(22, 22, 22, .88);
    border: 1px solid rgba(148,163,184,.2);
    border-radius: 18px;
    padding: 1.2rem 1.25rem;
    min-height: 180px;
}
.panel h3 { margin-top: 0; color: #f2f7ff; }
.muted { color: #a9bad2; font-size: .92rem; }
.metric {
    background: rgba(22,22,22,.88);
    border: 1px solid rgba(148,163,184,.2);
    border-radius: 15px;
    padding: 1rem;
}
.metric-label { color: #a9bad2; font-size: .8rem; }
.metric-value { color: #f8fbff; font-size: 1.25rem; font-weight: 700; margin-top: .25rem; }
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: .3rem .7rem;
    background: rgba(251,191,36,.12);
    color: #fcd34d;
    border: 1px solid rgba(251,191,36,.28);
    font-size: .78rem;
    font-weight: 700;
}
.small-note { color: #a9bad2; font-size: .82rem; }
div.stButton > button {
    border-radius: 10px;
    min-height: 2.7rem;
    font-weight: 650;
}
</style>
""", unsafe_allow_html=True)

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

with st.sidebar:
    st.markdown("## 🤟 SignBridge AI")
    st.caption("Accessible communication • Prototype")
    st.divider()
    st.markdown("### Project status")
    st.markdown('<span class="pill">MODEL INTEGRATION PENDING</span>', unsafe_allow_html=True)
    st.write("")
    st.markdown("**Current stack**")
    st.markdown("- Python")
    st.markdown("- MediaPipe + OpenCV")
    st.markdown("- scikit-learn")
    st.markdown("- Streamlit")
    st.divider()
    st.markdown("**Prototype scope**")
    st.caption("Limited-vocabulary static-sign recognition. This is not full ISL translation.")
    st.markdown("**Accessibility principle**")
    st.caption("Predictions should be visible, editable, and confirmed by the user.")

    st.markdown("""
    <div class="hero">
    <div class="hero-kicker">Computer vision for inclusive communication</div>
    <h1>SignBridge AI</h1>
    <p>A prototype exploring how hand-landmark detection and machine learning can support recognition of a limited set of Indian Sign Language signs.</p>
    </div>
    """, unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric"><div class="metric-label">Recognition engine</div><div class="metric-value">Being connected</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric"><div class="metric-label">Vocabulary</div><div class="metric-value">Dataset pending</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric"><div class="metric-label">Output mode</div><div class="metric-value">Text → Speech planned</div></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.05, 0.95], gap="medium")

with left:
    st.markdown(
        '<div class="panel"><h3>📷 Live sign recognition</h3>'
        '<p class="muted">Show one hand clearly with good lighting '
        'and keep your hand inside the camera frame.</p></div>',
        unsafe_allow_html=True
    )

    if model is None:
        st.error("Model not found. Check models/isl_sign_model.pkl")
    else:
        st.markdown("### Live sign recognition")

        webrtc_streamer(
            key="signbridge-camera",
            video_processor_factory=lambda: SignProcessor(prediction_queue),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
        )

    st.caption(
        "Hold a sign steady to recognize it. "
        "Briefly move both hands out of view before repeating the same sign."
    )

with right:
    st.markdown(
        '<div class="panel"><h3>💬 Sentence builder</h3>'
        '<p class="muted">Your recognized signs will appear here.</p></div>',
        unsafe_allow_html=True
    )

    if "sentence" not in st.session_state:
        st.session_state.sentence = ""

    st.session_state.sentence = st.text_area(
        "Text output",
        value=st.session_state.sentence,
        placeholder="Recognized signs will appear here…",
        height=115,
        help="You can edit the sentence manually."
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("↩ Undo last word", use_container_width=True):
            words = st.session_state.sentence.split()
            st.session_state.sentence = " ".join(words[:-1])
            st.rerun()

    with c2:
        if st.button("Clear text", use_container_width=True):
            st.session_state.sentence = ""
            st.rerun()

    if st.button("Copy sentence", use_container_width=True):
        sentence = st.session_state.sentence.strip()

        if sentence:
            st.code(sentence, language=None)
            st.caption("Select the text above and copy it.")
        else:
            st.info("Your sentence is empty.")

# Refresh the dashboard and display newly recognized words.
st_autorefresh(interval=1000, key="signbridge_refresh")

while not prediction_queue.empty():
    word = prediction_queue.get_nowait()
    current = st.session_state.sentence.strip()
    st.session_state.sentence = f"{current} {word}".strip()

st.write("")
st.markdown("### How it works")
steps = st.columns(4)
step_data = [
    ("01", "Image input", "An image or webcam frame"),
    ("02", "Landmarks", "MediaPipe detects hand keypoints"),
    ("03", "Classifier", "Model predicts a known label"),
    ("04", "Text output", "User reviews the prediction"),
]
for col, (num, title, desc) in zip(steps, step_data):
    with col:
        st.markdown(
            f'<div class="metric"><div class="metric-label">{num}</div><div class="metric-value">{title}</div><p class="small-note">{desc}</p></div>',
            unsafe_allow_html=True,
        )

st.info(
    "Responsible-use note: sign languages use hand movement, facial expression, body posture, and context. "
    "This prototype focuses on a limited set of static signs and may make mistakes; it should not be presented as a complete ISL translator."
)

st.markdown('<p class="small-note">SignBridge AI • Student open-source prototype • Dataset attribution will be added after dataset verification.</p>', unsafe_allow_html=True)


st.markdown("""
<style>
/* Background: charcoal with soft white light */
.stApp {
    background:
        radial-gradient(ellipse at 12% 12%,
            rgba(255,255,255,0.13), transparent 35%),
        radial-gradient(ellipse at 88% 35%,
            rgba(210,220,230,0.09), transparent 32%),
        linear-gradient(135deg, #101010, #1b1b1b 50%, #090909)
        !important;
    background-attachment: fixed !important;
}

/* Frosted-glass panels and cards */
.glass-panel,
.metric-card,
.hero,
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.075) !important;
    backdrop-filter: blur(22px) saturate(130%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(130%) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-top: 1px solid rgba(255,255,255,0.34) !important;
    border-radius: 20px !important;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.22),
        inset 0 1px 0 rgba(255,255,255,0.10) !important;
}

/* Sidebar: translucent glass */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.045) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255,255,255,0.14) !important;
}

/* White glass highlight on hover */
.glass-panel:hover,
.metric-card:hover {
    background: rgba(255,255,255,0.11) !important;
    border-color: rgba(255,255,255,0.30) !important;
    box-shadow: 0 12px 38px rgba(0,0,0,0.28),
                inset 0 1px 0 rgba(255,255,255,0.16) !important;
    transition: all 0.25s ease;
}

/* Inputs */
.stTextArea textarea,
.stTextInput input {
    background: rgba(255,255,255,0.07) !important;
    color: #f5f5f5 !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 14px !important;
}

/* Buttons */
.stButton button {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.24) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.2s ease;
}

.stButton button:hover {
    background: rgba(255,255,255,0.19) !important;
    border-color: rgba(255,255,255,0.48) !important;
}

/* Typography */
.stApp, [data-testid="stSidebar"] {
    color: #f4f4f4;
}

/* Avoid unwanted blue focus accents */
*:focus-visible {
    outline-color: rgba(255,255,255,0.65) !important;
}
</style>
""", unsafe_allow_html=True)

