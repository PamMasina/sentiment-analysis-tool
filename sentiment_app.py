import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
import time
from datetime import datetime
import requests
import re

# --- Page config ---
st.set_page_config(
    page_title="Sentiment Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark theme + custom styling ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #0f172a 0%, #020617 60%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background: #0a0f1e;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    h1, h2, h3, h4 {
        color: #f1f5f9 !important;
        letter-spacing: -0.02em;
    }
    .hero {
        padding: 0.5rem 0 1.5rem 0;
        animation: fadeIn 0.6s ease-out;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.1;
    }
    .hero-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #22d3ee;
        padding: 0.25rem 0.7rem;
        border: 1px solid #22d3ee55;
        border-radius: 999px;
        margin-bottom: 0.8rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.6rem;
    }
    .result-card {
        border-radius: 16px;
        padding: 1.6rem 1.4rem;
        background: #0b1224;
        border: 1px solid #1e293b;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: slideUp 0.5s ease-out;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-4px);
    }
    .result-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .card-positive::before { background: linear-gradient(90deg, #10b981, #22d3ee); box-shadow: 0 0 20px #10b981; }
    .card-negative::before { background: linear-gradient(90deg, #ef4444, #f472b6); box-shadow: 0 0 20px #ef4444; }
    .card-neutral::before  { background: linear-gradient(90deg, #64748b, #94a3b8); box-shadow: 0 0 20px #64748b; }
    .card-positive:hover { border-color: #10b981; }
    .card-negative:hover { border-color: #ef4444; }
    .card-neutral:hover  { border-color: #64748b; }
    .card-emoji {
        font-size: 3rem;
        line-height: 1;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 12px currentColor);
    }
    .card-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .card-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .card-positive .card-value { color: #34d399; text-shadow: 0 0 20px #10b98155; }
    .card-negative .card-value { color: #f87171; text-shadow: 0 0 20px #ef444455; }
    .card-neutral  .card-value { color: #cbd5e1; }
    .card-detail {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    .card-detail b { color: #e2e8f0; }
    .method-header {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.7rem;
        text-align: center;
    }
    .agreement {
        border-radius: 12px;
        padding: 1rem 1.4rem;
        text-align: center;
        font-weight: 600;
        margin-top: 1.5rem;
        animation: fadeIn 0.7s ease-out;
        border: 1px solid;
    }
    .agree-yes {
        background: linear-gradient(90deg, #10b98115, #22d3ee15);
        color: #34d399;
        border-color: #10b98155;
    }
    .agree-no {
        background: linear-gradient(90deg, #f59e0b15, #ef444415);
        color: #fbbf24;
        border-color: #f59e0b55;
    }
    .stButton > button {
        background: #0b1224;
        color: #e2e8f0;
        border: 1px solid #1e293b;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #22d3ee;
        color: #22d3ee;
        box-shadow: 0 0 20px #22d3ee33;
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        color: #020617;
        border: none;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 30px #22d3ee66;
        color: #020617;
    }
    .stTextArea textarea {
        background: #0b1224 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #22d3ee !important;
        box-shadow: 0 0 0 3px #22d3ee22 !important;
    }
    .sidebar-brand {
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sidebar-tag {
        font-size: 0.72rem;
        color: #64748b;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    .stButton > button[key="open_sidebar_btn"] {
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 999999 !important;
        width: 46px !important;
        height: 46px !important;
        padding: 0 !important;
        border-radius: 12px !important;
        background: #0b1224 !important;
        border: 1px solid #22d3ee !important;
        color: #22d3ee !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        box-shadow: 0 0 20px #22d3ee66 !important;
    }
    .stButton > button[key="open_sidebar_btn"]:hover {
        background: #22d3ee !important;
        color: #0b1224 !important;
        box-shadow: 0 0 30px #22d3eeaa !important;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stMetricValue"] { color: #f1f5f9; }
    [data-testid="stMetricLabel"] { color: #94a3b8; }
    hr { border-color: #1e293b !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    [data-testid="stDecoration"] {
        display: none;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .stDeployButton {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Load models ---
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

@st.cache_resource
def load_hf_model():
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")

hf_classifier = load_hf_model()

# --- Helpers ---
def vader_analyze(text):
    scores = sia.polarity_scores(text)
    c = scores['compound']
    if c >= 0.05: label = "Positive"
    elif c <= -0.05: label = "Negative"
    else: label = "Neutral"
    return label, c, scores

def hf_analyze(text):
    r = hf_classifier(text)[0]
    label = r['label'].capitalize()
    return label, r['score']

def emoji_for(label):
    return {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(label, "🤔")

def card_cls(label):
    return {"Positive": "card-positive", "Negative": "card-negative", "Neutral": "card-neutral"}.get(label, "card-neutral")

# --- Session state init ---
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# --- Sidebar toggle button (only shown when sidebar is CLOSED) ---
if not st.session_state.sidebar_open:
    if st.button("☰", key="open_sidebar_btn"):
        st.session_state.sidebar_open = True
        st.rerun()

# --- Sidebar ---
if st.session_state.sidebar_open:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🧠 Sentiment Lab</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tag">AI Analysis Suite</div>', unsafe_allow_html=True)

        if st.button("✕  Close sidebar", use_container_width=True, key="close_sidebar_btn"):
            st.session_state.sidebar_open = False
            st.rerun()

        page = st.radio(
            "Navigation",
            ["🔍 Analyze", "📊 Compare", "🕘 History", "📈 YouTube", "📖 About"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("**Model Stack**")
        st.markdown(
            "- 🟢 **VADER** — rule-based\n"
            "- 🔵 **DistilBERT** — deep learning\n"
            "- ⚡ **Streamlit** — interface"
        )

        st.markdown("---")
        st.caption("Built with Python 3.12")
else:
    page = "🔍 Analyze"

# --- Shared analysis function ---
def run_analysis(text):
    v_label, v_comp, v_scores = vader_analyze(text)
    h_label, h_conf = hf_analyze(text)
    return v_label, v_comp, v_scores, h_label, h_conf

# --- YouTube helpers ---
def get_youtube_comments(video_id, max_results=50):
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        return None, "API key not configured. Add YOUTUBE_API_KEY to .streamlit/secrets.toml"

    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": min(max_results, 100),
        "key": api_key,
        "textFormat": "plainText",
        "order": "relevance",
    }

    try:
        response = requests.get(base_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            comments = [
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in data.get("items", [])
            ]
            return comments, None
        elif response.status_code == 403:
            return None, "API quota exceeded or key restricted. Try again later."
        elif response.status_code == 404:
            return None, "Comments disabled for this video, or video not found."
        else:
            return None, f"YouTube API error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)[:60]}"


def extract_video_id(url):
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=|\&v=)([^#\&\?]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# =====================================================================
# PAGE: ANALYZE
# =====================================================================
if page == "🔍 Analyze":
    st.markdown("""
    <div class="hero">
        <span class="hero-tag">● Live Inference</span>
        <h1 class="hero-title">Sentiment Analysis</h1>
        <p class="hero-sub">Type any text. Two AI models analyze it in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    def fill_sample(txt):
        st.session_state.user_text = txt

    st.markdown("**Quick samples** — click to auto-fill")
    c1, c2, c3, c4 = st.columns(4)
    samples = [
        ("😊 Positive", "I absolutely love this product! Best purchase ever."),
        ("😞 Negative", "This is terrible. Complete waste of money."),
        ("😐 Neutral", "The meeting is scheduled for 3pm tomorrow."),
        ("🤔 Mixed", "Not bad, but not great either."),
    ]

    for col, (label, txt) in zip([c1, c2, c3, c4], samples):
        with col:
            st.button(
                label,
                use_container_width=True,
                key=f"btn_{label}",
                on_click=fill_sample,
                args=(txt,)
            )

    text_input = st.text_area(
        "Your text",
        key="user_text",
        placeholder="Type or paste something here...",
        height=130,
    )

    b1, b2, b3 = st.columns([1, 1, 1])
    with b2:
        analyze_clicked = st.button("⚡ Analyze", type="primary", use_container_width=True)

    if analyze_clicked and text_input.strip():
        v_label_tmp, v_comp_tmp, v_scores_tmp, h_label_tmp, h_conf_tmp = run_analysis(text_input)
        st.session_state.history.insert(0, {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Text": text_input,
            "VADER": v_label_tmp,
            "VADER Score": round(v_comp_tmp, 3),
            "Hugging Face": h_label_tmp,
            "HF Confidence": f"{h_conf_tmp*100:.1f}%",
            "Agree": "✅" if v_label_tmp == h_label_tmp else "⚠️",
        })
        st.session_state.history = st.session_state.history[:10]

    if text_input.strip():
        with st.spinner("Analyzing..."):
            v_label, v_comp, v_scores, h_label, h_conf = run_analysis(text_input)
            time.sleep(0.3)

        st.markdown("### Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="method-header">VADER · Rule-Based</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card {card_cls(v_label)}">
                <div class="card-emoji">{emoji_for(v_label)}</div>
                <div class="card-label">Sentiment</div>
                <div class="card-value">{v_label}</div>
                <div class="card-detail">Compound <b>{v_comp:+.3f}</b></div>
                <div class="card-detail">pos {v_scores['pos']:.2f} · neu {v_scores['neu']:.2f} · neg {v_scores['neg']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="method-header">DistilBERT · Deep Learning</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card {card_cls(h_label)}">
                <div class="card-emoji">{emoji_for(h_label)}</div>
                <div class="card-label">Sentiment</div>
                <div class="card-value">{h_label}</div>
                <div class="card-detail">Confidence <b>{h_conf*100:.1f}%</b></div>
                <div class="card-detail">Model distilbert-sst-2</div>
            </div>
            """, unsafe_allow_html=True)

        if v_label == h_label:
            st.markdown(f'<div class="agreement agree-yes">✓ Both models agree — <b>{v_label}</b></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="agreement agree-no">⚠ Models disagree — VADER: <b>{v_label}</b> · Hugging Face: <b>{h_label}</b></div>', unsafe_allow_html=True)

        st.markdown("### Visual Breakdown")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**VADER breakdown**")
            pie_fig = go.Figure(data=[go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[v_scores['pos'], v_scores['neu'], v_scores['neg']],
                hole=0.55,
                marker=dict(colors=["#10b981", "#64748b", "#ef4444"], line=dict(color="#0b1224", width=2)),
                textinfo="label+percent",
                textfont=dict(color="#e2e8f0", size=13),
                hovertemplate="<b>%{label}</b><br>Score: %{value:.2f}<br>%{percent}<extra></extra>"
            )])
            pie_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                showlegend=False,
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                annotations=[dict(text="VADER", x=0.5, y=0.5, font=dict(size=14, color="#94a3b8"), showarrow=False)]
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        with chart_col2:
            st.markdown("**Model confidence**")
            bar_fig = go.Figure(data=[
                go.Bar(
                    name="VADER",
                    x=["Positive", "Neutral", "Negative"],
                    y=[v_scores['pos'], v_scores['neu'], v_scores['neg']],
                    marker_color="#22d3ee",
                    hovertemplate="<b>VADER %{x}</b><br>%{y:.2f}<extra></extra>"
                ),
                go.Bar(
                    name="Hugging Face",
                    x=["Positive", "Negative"],
                    y=[
                        h_conf if h_label == "Positive" else 0,
                        h_conf if h_label == "Negative" else 0
                    ],
                    marker_color="#a78bfa",
                    hovertemplate="<b>HF %{x}</b><br>%{y:.2f}<extra></extra>"
                )
            ])
            bar_fig.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
                height=320,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", range=[0, 1])
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("**Hugging Face confidence gauge**")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=h_conf * 100,
            number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": "#a78bfa"},
                "bgcolor": "#0b1224",
                "borderwidth": 1,
                "bordercolor": "#1e293b",
                "steps": [
                    {"range": [0, 50], "color": "#1e293b"},
                    {"range": [50, 80], "color": "#334155"},
                    {"range": [80, 100], "color": "#475569"}
                ],
            },
            title={"text": f"Prediction: <b>{h_label}</b>", "font": {"color": "#94a3b8", "size": 14}}
        ))
        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=280,
            margin=dict(l=30, r=30, t=50, b=10)
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
    else:
        st.info("Enter some text to begin analysis.")

# =====================================================================
# PAGE: COMPARE
# =====================================================================
elif page == "📊 Compare":
    st.markdown("""
    <div class="hero">
        <span class="hero-tag">● Batch Mode</span>
        <h1 class="hero-title">Compare Models</h1>
        <p class="hero-sub">Paste multiple sentences (one per line) and see how both models perform.</p>
    </div>
    """, unsafe_allow_html=True)

    batch_text = st.text_area(
        "Sentences (one per line)",
        value="I love this product!\nThis is the worst experience.\nThe meeting is at 3pm.\nNot bad, but not great either.",
        height=180
    )

    if st.button("⚡ Run Batch Analysis", type="primary"):
        lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
        if lines:
            with st.spinner(f"Analyzing {len(lines)} sentences..."):
                rows = []
                for line in lines:
                    v_label, v_comp, _, h_label, h_conf = run_analysis(line)
                    rows.append({
                        "Text": line,
                        "VADER": v_label,
                        "VADER Score": round(v_comp, 3),
                        "Hugging Face": h_label,
                        "HF Confidence": f"{h_conf*100:.1f}%",
                        "Agree": "✅" if v_label == h_label else "⚠️"
                    })
                df = pd.DataFrame(rows)
            st.markdown("### Results")
            st.dataframe(df, use_container_width=True, hide_index=True)

            agree_count = sum(1 for r in rows if r["VADER"] == r["Hugging Face"])
            st.markdown(f"""
            <div class="agreement agree-yes" style="margin-top:1rem;">
                Agreement rate: <b>{agree_count}/{len(rows)} ({agree_count*100//len(rows)}%)</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please enter at least one sentence.")

# =====================================================================
# PAGE: HISTORY
# =====================================================================
elif page == "🕘 History":
    st.markdown("""
    <div class="hero">
        <span class="hero-tag">● Session Log</span>
        <h1 class="hero-title">Analysis History</h1>
        <p class="hero-sub">Your last 10 analyses from this session. Clears when you refresh.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No analyses yet. Head to **🔍 Analyze** and try some text.")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"**{len(st.session_state.history)}** of 10 entries stored")
        with col_b:
            if st.button("🗑 Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

        agree_count = sum(1 for r in st.session_state.history if r["Agree"] == "✅")
        total = len(st.session_state.history)
        if total > 0:
            st.markdown(f"""
            <div class="agreement agree-yes" style="margin-top:1rem;">
                Model agreement rate this session: <b>{agree_count}/{total} ({agree_count*100//total}%)</b>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# PAGE: YOUTUBE
# =====================================================================
elif page == "📈 YouTube":
    st.markdown("""
    <div class="hero">
        <span class="hero-tag">● API Mode</span>
        <h1 class="hero-title">YouTube Comments</h1>
        <p class="hero-sub">Paste a YouTube video link to analyze its comment section.</p>
    </div>
    """, unsafe_allow_html=True)

    video_url = st.text_input(
        "YouTube video URL",
        placeholder="e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    max_comments = st.slider("Number of comments to fetch", 10, 100, 50, step=10)

    if st.button("⚡ Fetch & Analyze Comments", type="primary"):
        if not video_url.strip():
            st.warning("Please paste a YouTube URL first.")
        else:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Couldn't find a video ID in that URL. Check the link and try again.")
            else:
                with st.spinner(f"Fetching up to {max_comments} comments..."):
                    comments, error = get_youtube_comments(video_id, max_comments)

                if error:
                    st.error(error)
                elif not comments:
                    st.info("No comments found for this video.")
                else:
                    st.success(f"Fetched **{len(comments)}** comments. Analyzing...")

                    with st.spinner("Running sentiment analysis on each comment..."):
                        vader_labels = []
                        hf_labels = []
                        rows = []
                        for c in comments:
                            v_label, v_comp, _, h_label, h_conf = run_analysis(c)
                            vader_labels.append(v_label)
                            hf_labels.append(h_label)
                            rows.append({
                                "Comment": c[:120] + ("..." if len(c) > 120 else ""),
                                "VADER": v_label,
                                "VADER Score": round(v_comp, 3),
                                "Hugging Face": h_label,
                                "HF Conf": f"{h_conf*100:.0f}%",
                            })

                    vader_counts = pd.Series(vader_labels).value_counts().to_dict()
                    total = len(comments)
                    pos = vader_counts.get("Positive", 0)
                    neu = vader_counts.get("Neutral", 0)
                    neg = vader_counts.get("Negative", 0)

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Total Comments", total)
                    kpi2.metric("😊 Positive", f"{pos} ({pos*100//total}%)")
                    kpi3.metric("😐 Neutral", f"{neu} ({neu*100//total}%)")
                    kpi4.metric("😞 Negative", f"{neg} ({neg*100//total}%)")

                    st.markdown("### Sentiment Distribution (VADER)")
                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        pie_fig = go.Figure(data=[go.Pie(
                            labels=["Positive", "Neutral", "Negative"],
                            values=[pos, neu, neg],
                            hole=0.55,
                            marker=dict(
                                colors=["#10b981", "#64748b", "#ef4444"],
                                line=dict(color="#0b1224", width=2)
                            ),
                            textinfo="label+percent",
                            textfont=dict(color="#e2e8f0", size=13),
                            hovertemplate="<b>%{label}</b><br>%{value} comments<br>%{percent}<extra></extra>"
                        )])
                        pie_fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2e8f0"),
                            showlegend=False,
                            height=340,
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        st.plotly_chart(pie_fig, use_container_width=True)

                    with chart_col2:
                        bar_fig = go.Figure(data=[go.Bar(
                            x=["Positive", "Neutral", "Negative"],
                            y=[pos, neu, neg],
                            marker_color=["#10b981", "#64748b", "#ef4444"],
                            hovertemplate="<b>%{x}</b><br>%{y} comments<extra></extra>"
                        )])
                        bar_fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2e8f0"),
                            height=340,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
                            yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)

                    st.markdown("### Analyzed Comments")
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

# =====================================================================
# PAGE: ABOUT
# =====================================================================
else:
    st.markdown("""
    <div class="hero">
        <span class="hero-tag">● Documentation</span>
        <h1 class="hero-title">About This Project</h1>
        <p class="hero-sub">A dual-model sentiment analyzer built for real-world text.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What It Does")
    st.markdown(
        "This tool analyzes the sentiment of any text using **two independent AI methods** "
        "and shows you where they agree and where they don't."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 VADER")
        st.markdown(
            "- Rule-based lexicon approach\n"
            "- Fast and lightweight\n"
            "- Handles three classes: Positive / Neutral / Negative\n"
            "- Best for short, informal text like social media"
        )
    with col2:
        st.markdown("#### 🔵 Hugging Face (DistilBERT)")
        st.markdown(
            "- Deep learning transformer model\n"
            "- Trained on SST-2 sentiment dataset\n"
            "- Binary output: Positive / Negative\n"
            "- Better at understanding context and nuance"
        )

    st.markdown("### Tech Stack")
    st.markdown(
        "- **Python 3.12**\n"
        "- **Streamlit** — web interface\n"
        "- **NLTK** — VADER sentiment analyzer\n"
        "- **Transformers + PyTorch** — Hugging Face model\n"
        "- **Pandas** — data handling\n"
        "- **Plotly** — interactive charts\n"
        "- **YouTube Data API v3** — comment fetching"
    )

    st.markdown("### Run It Locally")
    st.code("""# Clone the repo
git clone https://github.com/PamMasina/sentiment-analysis-tool.git
cd sentiment-analysis-tool

# Create and activate virtual environment
py -3.12 -m venv venv
venv\\Scripts\\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run sentiment_app.py""", language="powershell")