import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# --- VADER setup ---
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# --- Hugging Face setup (runs locally on your PC) ---
@st.cache_resource
def load_hf_model():
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")

hf_classifier = load_hf_model()

# --- App UI ---
st.set_page_config(page_title="Sentiment Analysis Tool", page_icon="💬")
st.title("💬 Sentiment Analysis Tool")
st.markdown("Enter text below to analyze its sentiment using **VADER** and **Hugging Face**.")

text_input = st.text_area("Enter your text:",
                          placeholder="Type something like 'I love this product!'",
                          height=150)

if st.button("Analyze Sentiment", type="primary"):
    if text_input.strip():
        # VADER
        vader_scores = sia.polarity_scores(text_input)
        vader_compound = vader_scores['compound']
        if vader_compound >= 0.05:
            vader_sentiment = "Positive"
        elif vader_compound <= -0.05:
            vader_sentiment = "Negative"
        else:
            vader_sentiment = "Neutral"

        # Hugging Face
        hf_result = hf_classifier(text_input)[0]

        # Display
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### VADER")
            st.metric("Sentiment", vader_sentiment)
            st.metric("Compound Score", f"{vader_compound:.3f}")
            st.caption(f"pos={vader_scores['pos']:.2f}, neu={vader_scores['neu']:.2f}, neg={vader_scores['neg']:.2f}")

        with col2:
            st.markdown("### Hugging Face")
            st.metric("Sentiment", hf_result['label'].capitalize())
            st.metric("Confidence", f"{hf_result['score']*100:.1f}%")
    else:
        st.warning("Please enter some text to analyze.")