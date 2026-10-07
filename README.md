# Sentiment Analysis Tool

A web app that analyzes text sentiment using two different approaches: **VADER** (rule-based) and **Hugging Face Transformers** (deep learning).

## Features

- Enter any text and get instant sentiment analysis
- Compare results from VADER and Hugging Face side by side
- Classifies text as **Positive**, **Neutral**, or **Negative**

## Tech Stack

- **Python 3.12**
- **Streamlit** — web interface
- **NLTK (VADER)** — rule-based sentiment analysis
- **Hugging Face Transformers** — deep learning model (`distilbert-base-uncased-finetuned-sst-2-english`)
- **PyTorch** — backend for the Hugging Face model

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/PamMasina/sentiment-analysis-tool.git
cd sentiment-analysis-tool

# Create and activate a virtual environment
py -3.12 -m venv venv
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run sentiment_app.py