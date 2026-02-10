# Sentiment Analysis of Indonesian App Reviews

## 📌 Project Overview
This project performs **sentiment analysis on Indonesian-language application reviews** scraped from the Google Play Store. The goal is to classify user reviews into **Positive, Negative, and Neutral** sentiments using both **Machine Learning** and **Deep Learning** approaches.

Unlike simple rating-based labeling, this project applies **lexicon-based sentiment labeling using the INSET Indonesian Sentiment Lexicon**, enhanced with proper text preprocessing. This makes the sentiment labels more representative of the actual opinion expressed in the text.

This project was originally developed as part of the **Dicoding – Belajar Fundamental Deep Learning** course and has been further polished for portfolio and real-world applicability.

---

## 🎯 Objectives
- Scrape real-world Indonesian app reviews from Google Play Store
- Perform robust text preprocessing for Indonesian NLP
- Apply **lexicon-based sentiment labeling** (INSET)
- Train and compare multiple models:
  - TF-IDF + Logistic Regression
  - TF-IDF + Support Vector Machine (SVM)
  - LSTM (Deep Learning)
- Evaluate model performance and perform inference on unseen text

---

## 🗂️ Project Structure
```
sentiment-analysis-indonesian-app-review/
│
├── data/
│   ├── raw_reviews.csv            # Raw scraped reviews
│   ├── labeled_reviews.csv        # Cleaned & labeled dataset
│   └── lexicon/
│       ├── positive.tsv           # INSET positive lexicon
│       └── negative.tsv           # INSET negative lexicon
│
├── notebooks/
│   ├── 01_scraping.ipynb
│   ├── 02_preprocessing_labeling.ipynb
│   ├── 03_modeling_ml.ipynb
│   └── 04_modeling_lstm.ipynb
│
├── models/
│   ├── svm_model.pkl
│   └── lstm_model.h5
│
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset
- **Source**: Google Play Store (scraped manually)
- **Language**: Indonesian
- **Classes**: Positive, Negative, Neutral
- **Labeling Method**: Lexicon-based sentiment analysis using **INSET Bahasa Indonesia**

Sentiment distribution after labeling:
- Positive: ~60%
- Negative: ~28%
- Neutral: ~12%

---

## 🧹 Text Preprocessing
Key preprocessing steps include:
- Case folding
- URL, number, and punctuation removal
- Stopword removal (Indonesian)
- Tokenization

Preprocessing is applied **before sentiment labeling** to ensure lexicon matching accuracy.

---

## 🤖 Models & Results
| Model | Accuracy |
|------|----------|
| TF-IDF + Logistic Regression | 87.0% |
| TF-IDF + SVM | **88.4%** |
| LSTM (Deep Learning) | 87.0% |

The SVM model achieved the highest performance and is recommended for deployment.

---

## 🔍 Inference Example
```text
Input: "aplikasinya sering error dan makin mahal"
Output: Negative
```

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- NLTK
- Google Play Scraper

---

## 🚀 Future Improvements
- Negation handling for lexicon-based labeling
- Confusion matrix visualization
- Model deployment using Streamlit or Flask
- Domain-specific lexicon expansion

---

## 👤 Author
**Yusuf Putra Bintang Satria**  
Informatics Engineering Graduate  
Focus: Data Science, Machine Learning, NLP

---

## 📄 License
This project is for educational and portfolio purposes.
