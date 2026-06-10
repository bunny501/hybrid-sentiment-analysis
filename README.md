# Hybrid Sentiment Analysis Framework

An end-to-end Natural Language Processing (NLP) project for sentiment classification using TF-IDF feature engineering, Logistic Regression, model evaluation, and interactive analytics.

## Overview

This project analyzes textual reviews and predicts whether the sentiment is **Positive** or **Negative**. The system implements a complete machine learning pipeline including text preprocessing, feature extraction, model training, evaluation, and visualization.

The project was developed to demonstrate practical applications of Natural Language Processing, Machine Learning, and Data Analysis using Python and Scikit-Learn.

---

## Features

* Text preprocessing and cleaning
* Stopword removal with negation preservation
* Porter Stemming
* TF-IDF vectorization (Unigrams + Bigrams)
* Logistic Regression classifier
* 5-Fold Stratified Cross Validation
* ROC-AUC evaluation
* Confusion Matrix analysis
* Feature Importance extraction
* Interactive dashboard visualizations
* Real-time sentiment prediction on unseen reviews

---

## Dataset

The project uses a combined dataset containing:

* IMDb Movie Reviews
* Amazon Product Reviews

Dataset includes both positive and negative sentiment labels.

---

## Machine Learning Pipeline

Raw Review Text
→ Text Cleaning
→ Stopword Removal
→ Stemming
→ TF-IDF Feature Extraction
→ Logistic Regression
→ Evaluation & Visualization

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Matplotlib
* Seaborn

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix
* Cross Validation

---

## Project Structure

```text
Sentiment_Project/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── sentiment_analyzer.py
│   ├── sentiment_main.py
│   ├── model_comparison.py
│   ├── hyperparameter_tuning.py
│   └── error_analysis.py
│
├── app/
│   └── streamlit_app.py
│
└── data/
    └── reviews.csv
```

## Installation

Clone the repository:

```bash
git clone https://github.com/bunny501/hybrid-sentiment-analysis.git
cd hybrid-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/sentiment_main.py
```

---

## Sample Predictions

| Review                                       | Prediction             |
| -------------------------------------------- | ---------------------- |
| Absolutely loved every minute of this movie. | Positive               |
| Worst purchase I have ever made.             | Negative               |
| Decent product but slightly overpriced.      | Mixed / Model Decision |

---

## Future Improvements

* Linear SVM benchmarking
* Naive Bayes benchmarking
* Hyperparameter tuning with GridSearchCV
* Model persistence using Joblib
* Streamlit deployment
* Larger review datasets (IMDb 50K Reviews)
* Transformer-based models (BERT)

---

## Learning Outcomes

Through this project I gained hands-on experience with:

* Natural Language Processing
* Text Feature Engineering
* Machine Learning Pipelines
* Model Evaluation
* Data Visualization
* Cross Validation Techniques
* Sentiment Analysis Systems

---

## Author

Rakesh Sainatha Reddy Bandi

B.Tech Computer Science (AI)

GitHub: https://github.com/bunny501
