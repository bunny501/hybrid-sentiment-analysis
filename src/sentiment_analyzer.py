"""
Sentiment Analysis Engine
Classifies reviews as positive/negative using TF-IDF + Logistic Regression.
Supports IMDB and Amazon review datasets.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    End-to-end Sentiment Analysis Pipeline.

    Steps:
    1. Text preprocessing  — clean, normalize, stem
    2. TF-IDF vectorization — convert text to feature matrix
    3. Logistic Regression  — binary classifier (positive / negative)
    4. Evaluation           — accuracy, AUC-ROC, confusion matrix, CV scores
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()

        # negation words we keep despite being stopwords
        self.keep_words = {'not', 'no', 'never', 'nor', "n't", 'nothing',
                           'nobody', 'nowhere', 'neither', 'without'}
        self.stop_words -= self.keep_words

        self.pipeline = None
        self.tfidf = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

        # stored for reporting
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred = self.y_prob = None
        self.feature_names = None

    # ─────────────────────────────────────────────
    # Step 1: Text Preprocessing
    # ─────────────────────────────────────────────
    def preprocess(self, text: str) -> str:
        """
        Clean and normalize a review string:
        - Lowercase
        - Remove HTML tags
        - Remove URLs and emails
        - Remove special characters (keep apostrophes for contractions)
        - Tokenize, remove stopwords, apply Porter stemming
        """
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)           # HTML tags
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
        text = re.sub(r'\S+@\S+', '', text)             # Emails
        text = re.sub(r"[^a-z\s']", ' ', text)          # Keep letters + apostrophes
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return ' '.join(tokens)

    def preprocess_batch(self, texts) -> list:
        return [self.preprocess(t) for t in texts]

    # ─────────────────────────────────────────────
    # Step 2 & 3: Build Pipeline & Train
    # ─────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, text_col='review', label_col='label',
            test_size=0.2, random_state=42):
        """
        Preprocess → TF-IDF → Logistic Regression.
        Splits data, trains the pipeline, stores evaluation artifacts.
        """
        # Preprocessing
        print("  [1/4] Preprocessing text...")
        df = df.copy()
        df['clean'] = self.preprocess_batch(df[text_col])

        X = df['clean'].values
        y = df[label_col].values

        # Train / test split (stratified)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te

        # TF-IDF
        print("  [2/4] Building TF-IDF matrix...")
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,       # apply log normalization
            min_df=1,
            max_df=0.95,
            analyzer='word',
        )

        # Logistic Regression
        self.model = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        )

        # Sklearn Pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('clf',   self.model),
        ])

        print("  [3/4] Training Logistic Regression...")
        self.pipeline.fit(X_tr, y_tr)
        self.feature_names = self.tfidf.get_feature_names_out()
        self.is_fitted = True

        # Evaluate
        print("  [4/4] Evaluating model...")
        self.y_pred = self.pipeline.predict(X_te)
        self.y_prob = self.pipeline.predict_proba(X_te)[:, 1]
        return self

    # ─────────────────────────────────────────────
    # Step 4: Evaluation Utilities
    # ─────────────────────────────────────────────
    def get_metrics(self) -> dict:
        acc  = accuracy_score(self.y_test, self.y_pred)
        auc  = roc_auc_score(self.y_test, self.y_prob)
        cm   = confusion_matrix(self.y_test, self.y_pred)
        rep  = classification_report(
            self.y_test, self.y_pred,
            target_names=['Negative', 'Positive'], output_dict=True
        )
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)

        return {
            'accuracy':   round(acc, 4),
            'auc_roc':    round(auc, 4),
            'confusion_matrix': cm,
            'report':     rep,
            'roc_curve':  (fpr, tpr, thresholds),
        }

    def cross_validate(self, df: pd.DataFrame, text_col='review',
                       label_col='label', cv=5) -> dict:
        """5-fold stratified cross-validation."""
        clean = self.preprocess_batch(df[text_col])
        y = df[label_col].values
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True, min_df=1, max_df=0.95
            )),
            ('clf', LogisticRegression(
                C=1.0, solver='lbfgs', max_iter=1000,
                class_weight='balanced', random_state=42
            )),
        ])
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, clean, y, cv=skf, scoring='accuracy')
        return {
            'scores': scores,
            'mean':   round(scores.mean(), 4),
            'std':    round(scores.std(),  4),
        }

    def get_top_features(self, n=20) -> dict:
        """Top positive and negative weighted TF-IDF features."""
        coef = self.model.coef_[0]
        top_pos_idx = np.argsort(coef)[-n:][::-1]
        top_neg_idx = np.argsort(coef)[:n]
        return {
            'positive': [(self.feature_names[i], round(coef[i], 4)) for i in top_pos_idx],
            'negative': [(self.feature_names[i], round(coef[i], 4)) for i in top_neg_idx],
        }

    def get_tfidf_stats(self) -> dict:
        X_tr = self.tfidf.transform(self.X_train)
        return {
            'vocab_size':       len(self.feature_names),
            'train_shape':      X_tr.shape,
            'test_shape':       self.tfidf.transform(self.X_test).shape,
            'sparsity':         round(1 - X_tr.nnz / (X_tr.shape[0] * X_tr.shape[1]), 4),
            'avg_nonzero_feats': round(X_tr.nnz / X_tr.shape[0], 2),
        }

    # ─────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────
    def predict(self, texts) -> list:
        """Predict sentiment for a list of raw review strings."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        clean = self.preprocess_batch(texts if isinstance(texts, list) else [texts])
        preds = self.pipeline.predict(clean)
        probs = self.pipeline.predict_proba(clean)[:, 1]
        return [
            {
                'text':      t[:80] + '...' if len(t) > 80 else t,
                'sentiment': 'positive' if p == 1 else 'negative',
                'confidence': round(float(pr if p == 1 else 1 - pr), 4),
                'prob_positive': round(float(pr), 4),
            }
            for t, p, pr in zip(texts if isinstance(texts, list) else [texts], preds, probs)
        ]
