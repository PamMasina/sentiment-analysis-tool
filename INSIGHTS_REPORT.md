# Sentiment Analysis Tool — Data Insights Report

**Author:** Andile Pamela Masina  
**Project:** Week 3 — Sentiment Analysis Tool  
**Bootcamp:** CAPACITI AI Bootcamp  
**Date:**7 October 2026

---

## 1. Objective

To evaluate how two different sentiment analysis approaches perform on the same input text:

- **VADER** — a rule-based lexicon method (NLTK)
- **Hugging Face Transformers** — a deep learning model (DistilBERT fine-tuned on SST-2)

The goal was to understand when each approach is more appropriate and how they differ in classification and confidence.

---

## 2. Test Cases and Results

| # | Input Text | VADER Result | VADER Score | Hugging Face Result | HF Confidence |
|---|------------|--------------|-------------|---------------------|---------------|
| 1 | I love this | Positive | 0.637 | Positive | 100.0% |
| 2 | This is the worst thing ever | Negative | -0.625 | Negative | 100.0% |
| 3 | The meeting is at 3pm | Neutral | 0.000 | Positive | 94.7% |
| 4 | I'm not sure how I feel about this | Negative | -0.241 | Negative | 99.9% |
| 5 | Best purchase I've made all year! | Positive | 0.670 | Positive | 100.0% |
| 6 | It's okay, nothing special | Negative | -0.092 | Negative | 80.8% |
| 7 | Absolutely fantastic service! | Positive | 0.635 | Positive | 100.0% |
| 8 | I hate waiting in long queues | Negative | -0.572 | Negative | 99.7% |
| 9 | The product arrived on time | Neutral | 0.000 | Positive | 98.4% |
| 10 | Not bad, but not great either | Negative | -0.545 | Negative | 98.8% |


---

## 3. Key Observations


- VADER correctly identifies **Neutral** text, while Hugging Face tends to force a Positive/Negative label because it's a binary model.
- Hugging Face is **more confident** on strongly emotional text.
- VADER is faster because it doesn't require a neural network.
- Hugging Face handles **negation and context** better (e.g., "not bad").
- For short, slang-heavy text (like tweets), VADER often performs well.

---

## 4. Conclusion


Combining VADER and Hugging Face provides a more complete picture than either alone. VADER is ideal for quick, explainable sentiment classification, while Hugging Face excels at capturing nuance and context. For production use, the choice depends on the data type and speed requirements.

---

## 5. Technical Notes

- **Model used:** `distilbert-base-uncased-finetuned-sst-2-english`
- **VADER lexicon:** `vader_lexicon` (NLTK)
- **Framework:** Streamlit (web UI)
- **Language:** Python 3.12