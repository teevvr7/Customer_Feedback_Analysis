# Customer Feedback Analysis 🚀

An end-to-end Natural Language Processing (NLP) pipeline designed for sentiment analysis on customer feedback and product reviews. This repository is built both as a practical utility and as an educational reference demonstrating every phase of the traditional NLP lifecycle—from raw text ingestion, cleaning, and feature engineering to model training, evaluation, and production API serving.

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Architecture & NLP Pipeline](#architecture--nlp-pipeline)
  - [1. Data Ingestion & Automated Labeling (VADER)](#1-data-ingestion--automated-labeling-vader)
  - [2. Text Preprocessing & Cleaning (spaCy + Regex)](#2-text-preprocessing--cleaning-spacy--regex)
  - [3. Feature Extraction (TF-IDF Vectorization)](#3-feature-extraction-tf-idf-vectorization)
  - [4. Model Training (Logistic Regression)](#4-model-training-logistic-regression)
  - [5. Model Evaluation & Confusion Matrix](#5-model-evaluation--confusion-matrix)
  - [6. Serving & Deployment (FastAPI)](#6-serving--deployment-fastapi)
- [Repository Structure](#repository-structure)
- [Educational Insights & Best Practices](#educational-insights--best-practices)
- [Installation & Setup](#installation--setup)
- [Step-by-Step Execution Guide](#step-by-step-execution-guide)
- [API Usage](#api-usage)
- [License](#license)

---

## Project Overview

In real-world applications, customer feedback (reviews, survey responses, social mentions) is unstructured, noisy, and unlabelled. Analyzing this data automatically allows organizations to monitor customer satisfaction and flag product issues in real time.

This project demonstrates how to:
1. Label unstructured text automatically using lexicon-based weak supervision.
2. Clean noisy text while preserving critical semantic context (e.g., negations).
3. Transform text into numerical matrices using Term Frequency-Inverse Document Frequency (TF-IDF) with n-grams.
4. Train and evaluate an interpretable classifier that handles class imbalance.
5. Deploy the trained pipeline as a lightweight, low-latency REST API using FastAPI.

---

## Architecture & NLP Pipeline

```
Raw Review Text (data/raw/Reviews.csv)
       │
       ▼
[ Step 1: Automated Sentiment Labeling ]
       ├─ NLTK VADER Lexicon Analyzer
       └─ Generates: positive, neutral, negative
       │
       ▼
[ Step 2: Text Preprocessing & Cleaning ]
       ├─ Lowercasing & Regex URL / Special Character Removal
       ├─ spaCy Tokenization & Lemmatization
       └─ Negation Preservation (keep "not", "no", "never", etc.)
       │
       ▼
[ Step 3: Feature Engineering (TF-IDF) ]
       ├─ Unigrams + Bigrams (ngram_range=(1, 2))
       ├─ Max Features = 10,000
       └─ Output: Sparse Matrix X
       │
       ▼
[ Step 4: Classification & Modeling ]
       ├─ Stratified Train/Test Split (80/20)
       ├─ Logistic Regression with Balanced Class Weights
       └─ Persistence: sentiment_model.pkl & tfidf_vectorizer.pkl
       │
       ▼
[ Step 5: Evaluation & Serving ]
       ├─ Classification Report & Confusion Matrix
       ├─ Interactive Terminal Predictor (CLI)
       └─ FastAPI Microservice (POST /predict)
```

---

### 1. Data Ingestion & Automated Labeling (VADER)
- **Module**: `src/data/preprocess.py` (`create_sentiment`)
- **Concept**: Weak supervision / Lexicon-based heuristic labeling.
- **How it works**:
  When unlabelled text datasets are available, **NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)** computes polarity scores:
  - **Positive**: Compound score $\ge 0.05$
  - **Negative**: Compound score $\le -0.05$
  - **Neutral**: Compound score between $-0.05$ and $0.05$

### 2. Text Preprocessing & Cleaning (spaCy + Regex)
- **Module**: `src/data/preprocess.py` (`clean_text`, `clean_batch`)
- **Key Concepts**:
  - **Vectorized String Filtering**: URLs (`http\S+`) and non-alphabetic characters are stripped using regular expressions.
  - **Lemmatization**: spaCy reduces inflected words to their dictionary root form (e.g., *"running"*, *"ran"* $\rightarrow$ *"run"*).
  - **Crucial Lesson — Negation Preservation**: Standard stop-word removal strips words like *"not"*, *"never"*, and *"no"*. In sentiment analysis, dropping negations flips sentiment completely (*"not good"* $\rightarrow$ *"good"*). Our pipeline protects negations using an explicit whitelist (`KEEP = {"not", "no", "never", "nor", "none", "nt"}`).
  - **Batch Performance**: Uses `nlp.pipe()` with `disable=["parser", "ner"]` to bypass CPU-heavy syntactic parsing and named-entity recognition when only tokenization/lemmatization is needed.

### 3. Feature Extraction (TF-IDF Vectorization)
- **Module**: `src/features/vectorizer.py`
- **Key Concepts**:
  - **TF (Term Frequency)**: Frequency of a term in a document.
  - **IDF (Inverse Document Frequency)**: Down-weights common words appearing across all documents.
  - **N-Grams (`ngram_range=(1, 2)`)**: Includes both single words (unigrams) and pairs of adjacent words (bigrams). This allows the vectorizer to preserve phrases like `"not recommend"` or `"great quality"`.
  - **Dimensionality Control (`max_features=10000`)**: Caps the vocabulary to the top 10,000 most predictive features to prevent sparse matrix bloat and overfitting.

### 4. Model Training (Logistic Regression)
- **Module**: `src/models/train.py`
- **Key Concepts**:
  - **Algorithm**: Logistic Regression with `max_iter=1000`.
  - **Class Imbalance Handling**: E-commerce customer reviews are heavily skewed towards positive ratings. By specifying `class_weight="balanced"`, the model penalizes misclassifications in minority classes (negative/neutral) proportionally higher.
  - **Reproducibility**: `random_state=42` ensures consistent train/test splits.

### 5. Model Evaluation & Confusion Matrix
- **Module**: `src/models/evaluate.py`
- **Metrics Evaluated**:
  - **Precision**: How many of the predicted instances for a class were truly correct.
  - **Recall**: How many of the actual instances of a class were captured by the model.
  - **F1-Score**: The harmonic mean of precision and recall.
  - **Confusion Matrix**: Fixed label ordering (`negative`, `neutral`, `positive`) to inspect cross-class confusion.

**Sample Evaluation Output:**
```
              precision    recall  f1-score   support

    negative       0.47      0.52      0.49        29
     neutral       0.57      0.33      0.42        12
    positive       0.93      0.93      0.93       279

    accuracy                           0.87       320
   macro avg       0.66      0.59      0.61       320
weighted avg       0.87      0.87      0.87       320
```

### 6. Serving & Deployment (FastAPI)
- **Module**: `src/api/app.py`
- **Key Concepts**:
  - Serves predictions over HTTP using FastAPI and Pydantic schemas.
  - Loads serialized artifacts (`models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`) into memory once on startup for sub-millisecond inference latency.

---

## Repository Structure

```
Customer_Feedback_Analysis/
├── LICENSE                         # MIT License
├── README.md                        # Documentation & Learning Guide
├── requirements.txt                 # Python dependencies
├── main.py                          # Pipeline entrypoint for data processing
├── data/
│   ├── raw/                         # Raw input datasets (Reviews.csv)
│   ├── processed/                   # Cleaned and sentiment-tagged reviews
│   └── sample/                      # Sample datasets for inference testing
├── models/
│   ├── sentiment_model.pkl          # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer artifact
└── src/
    ├── api/
    │   └── app.py                   # FastAPI REST API endpoint
    ├── data/
    │   ├── load_data.py             # Data loading utilities
    │   ├── preprocess.py            # Cleaning, lemmatization & VADER sentiment tagging
    │   └── process_dataset.py       # Full dataset processing runner
    ├── features/
    │   └── vectorizer.py            # TF-IDF feature extraction script
    └── models/
        ├── train.py                 # Model training & serialization script
        ├── evaluate.py              # Performance evaluation & confusion matrix
        └── predict.py               # Interactive CLI for real-time predictions
```

---

## Educational Insights & Best Practices

1. **Why serialize the Vectorizer alongside the Model?**
   At inference time, raw input text must be converted into numerical features using the exact same vocabulary index and IDF weights learned during training. Always save and load both `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.
2. **Lexicon vs. Machine Learning:**
   - *Lexicon (VADER)*: Rule-based, zero training required, fast, but lacks domain customization.
   - *Machine Learning (TF-IDF + Classifier)*: Learns domain-specific associations (e.g., product defects or specific praise) from data distributions.
3. **Handling Stop Words Intelligently:**
   Never blindly remove all standard stop words in sentiment tasks. Dropping negative qualifiers will distort the true sentiment.
4. **Addressing Real-world Class Imbalance:**
   Accuracy alone is misleading on imbalanced datasets. Look at Macro F1 and minority-class recall.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/teevvr7/Customer_Feedback_Analysis.git
cd Customer_Feedback_Analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Language Model & NLTK Lexicon
```bash
python -m spacy download en_core_web_sm
```

---

## Step-by-Step Execution Guide

### 1. Process & Clean Dataset
Run the data processing pipeline to label raw reviews with VADER and clean text with spaCy:
```bash
python main.py
```
*Output generated: `data/processed/cleaned_reviews.csv`*

### 2. Train the Sentiment Model
Extract TF-IDF features and train the Logistic Regression classifier:
```bash
python -m src.models.train
```
*Artifacts generated: `models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`*

### 3. Evaluate the Model
Generate the classification report and confusion matrix:
```bash
python -m src.models.evaluate
```

### 4. Run Interactive CLI Predictor
Test custom review phrases in your terminal:
```bash
python -m src.models.predict
```
*Example:*
```
Enter Review: The screen quality is crisp and battery lasts for weeks!
Prediction: positive

Enter Review: Extremely slow and freezes constantly. Would not recommend.
Prediction: negative
```

---

## API Usage

### Start the FastAPI Server
```bash
uvicorn src.api.app:app --reload
```
The API documentation (Swagger UI) will be accessible at: `http://127.0.0.1:8000/docs`

### Predict Endpoint (`POST /predict`)

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The battery life is amazing and reading at night is so comfortable."}'
```

**Response:**
```json
{
  "sentiment": "positive"
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
