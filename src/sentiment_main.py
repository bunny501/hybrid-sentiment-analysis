"""
Main Analysis Script — Sentiment Analysis System
IMDB + Amazon Reviews | TF-IDF | Logistic Regression
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sentiment_analyzer import SentimentAnalyzer
import os, warnings
warnings.filterwarnings('ignore')

# ── Theme ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#21262d',
    'grid.linestyle': '--', 'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans', 'axes.titlecolor': '#f0f6fc',
})
BG   = '#0d1117'; PANEL = '#161b22'; CARD = '#21262d'
BLUE = '#58a6ff'; GREEN = '#3fb950'; RED = '#f78166'
PURP = '#d2a8ff'; GOLD = '#e3b341'; TEAL = '#39d353'

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  SENTIMENT ANALYSIS SYSTEM")
print("  IMDB + Amazon Reviews | TF-IDF | Logistic Regression")
print("=" * 62)

# Load data
df = pd.read_csv('reviews.csv')
print(f"\n[1/7] Dataset loaded: {df.shape[0]} reviews")
print(f"      IMDB: {len(df[df.source=='IMDB'])}  |  Amazon: {len(df[df.source=='Amazon'])}")
print(f"      Positive: {df.label.sum()}  |  Negative: {(df.label==0).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Training model...")
sa = SentimentAnalyzer(max_features=3000, ngram_range=(1, 2))
sa.fit(df)

metrics = sa.get_metrics()
tfidf_s = sa.get_tfidf_stats()
top_f   = sa.get_top_features(n=15)

print(f"\n  Accuracy  : {metrics['accuracy']*100:.2f}%")
print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
print(f"  Vocab size: {tfidf_s['vocab_size']} features")
print(f"  Sparsity  : {tfidf_s['sparsity']*100:.1f}%")
print(f"  Train size: {tfidf_s['train_shape']}  |  Test size: {tfidf_s['test_shape']}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Cross-validation...")
cv = sa.cross_validate(df, cv=5)
print(f"  5-Fold CV scores: {[round(s,4) for s in cv['scores']]}")
print(f"  Mean ± Std: {cv['mean']*100:.2f}% ± {cv['std']*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Classification report...")
rep = metrics['report']
print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
print("  " + "─"*48)
for cls in ['Negative', 'Positive']:
    r = rep[cls]
    print(f"  {cls:<12} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1-score']:>8.3f} {int(r['support']):>9}")
print(f"  {'Macro avg':<12} {rep['macro avg']['precision']:>10.3f} {rep['macro avg']['recall']:>8.3f} {rep['macro avg']['f1-score']:>8.3f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Top predictive features...")
print("\n  TOP POSITIVE WORDS:")
for w, s in top_f['positive'][:10]:
    print(f"    {w:<30} coef = {s:+.4f}")
print("\n  TOP NEGATIVE WORDS:")
for w, s in top_f['negative'][:10]:
    print(f"    {w:<30} coef = {s:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Live inference on new reviews...")
test_reviews = [
    "Absolutely loved every minute of this. A stunning and deeply moving experience.",
    "Total garbage. Fell apart the same day I bought it. Do not waste your money.",
    "Decent product but the quality could be better for the price. Average experience.",
    "One of the best movies I have ever seen. Brilliantly acted and beautifully shot.",
    "Arrived broken and customer service refused to help. Completely unacceptable.",
    "Not what I expected at all. Misleading description and poor build quality.",
    "Masterful storytelling. The performances were incredible and the ending was perfect.",
    "Nothing special. Does the job but nothing about it stands out in any way.",
]
predictions = sa.predict(test_reviews)
print(f"\n  {'Review (truncated)':<55} {'Sentiment':<12} {'Confidence'}")
print("  " + "─"*80)
for p in predictions:
    emoji = "✅" if p['sentiment'] == 'positive' else "❌"
    print(f"  {p['text']:<55} {emoji} {p['sentiment']:<10} {p['confidence']*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Generating visualizations...")

# ══ Figure 1: EDA Dashboard ═══════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle('Sentiment Analysis System — EDA & Model Dashboard',
             fontsize=18, color='#f0f6fc', fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Sentiment distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = df['sentiment'].value_counts()
colors_bar = [GREEN, RED]
bars = ax1.bar(counts.index, counts.values, color=colors_bar, edgecolor='none', width=0.5)
ax1.set_title('Sentiment Distribution', fontsize=12, color='#f0f6fc', pad=8)
ax1.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, val+0.5, str(val),
             ha='center', va='bottom', fontsize=11, color='#e6edf3', fontweight='bold')
ax1.set_facecolor(CARD); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# 2. Source distribution
ax2 = fig.add_subplot(gs[0, 1])
src = df.groupby(['source','sentiment']).size().unstack()
x = np.arange(len(src))
w = 0.35
ax2.bar(x - w/2, src['positive'], w, color=GREEN, label='Positive', edgecolor='none', alpha=0.9)
ax2.bar(x + w/2, src['negative'], w, color=RED,   label='Negative', edgecolor='none', alpha=0.9)
ax2.set_xticks(x); ax2.set_xticklabels(src.index)
ax2.set_title('Reviews by Source & Sentiment', fontsize=12, color='#f0f6fc', pad=8)
ax2.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
ax2.set_facecolor(CARD); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# 3. Review length distribution
ax3 = fig.add_subplot(gs[0, 2])
df['word_count'] = df['review'].str.split().str.len()
for sentiment, color, label in [('positive', GREEN, 'Positive'), ('negative', RED, 'Negative')]:
    sub = df[df['sentiment'] == sentiment]['word_count']
    ax3.hist(sub, bins=15, alpha=0.7, color=color, label=label, edgecolor='none')
ax3.set_title('Word Count Distribution', fontsize=12, color='#f0f6fc', pad=8)
ax3.set_xlabel('Words per Review'); ax3.set_ylabel('Count')
ax3.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
ax3.set_facecolor(CARD); ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

# 4. Confusion Matrix
ax4 = fig.add_subplot(gs[1, 0])
cm = metrics['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'],
            cbar_kws={'shrink': 0.8})
ax4.set_title('Confusion Matrix', fontsize=12, color='#f0f6fc', pad=8)
ax4.set_xlabel('Predicted'); ax4.set_ylabel('Actual')
ax4.set_facecolor(CARD)

# 5. ROC Curve
ax5 = fig.add_subplot(gs[1, 1])
fpr, tpr, _ = metrics['roc_curve']
ax5.plot(fpr, tpr, color=BLUE, linewidth=2.5,
         label=f'AUC = {metrics["auc_roc"]:.3f}')
ax5.plot([0,1],[0,1], color='#8b949e', linestyle='--', linewidth=1)
ax5.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax5.set_title('ROC Curve', fontsize=12, color='#f0f6fc', pad=8)
ax5.set_xlabel('False Positive Rate'); ax5.set_ylabel('True Positive Rate')
ax5.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor='#e6edf3')
ax5.set_facecolor(CARD); ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

# 6. Stats card
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off'); ax6.set_facecolor(CARD)
ax6.set_title('Model Performance', fontsize=12, color='#f0f6fc', pad=8)
stat_items = [
    ("Accuracy",         f"{metrics['accuracy']*100:.2f}%",  BLUE),
    ("AUC-ROC",          f"{metrics['auc_roc']:.4f}",         GREEN),
    ("CV Mean",          f"{cv['mean']*100:.2f}%",             PURP),
    ("CV Std",           f"±{cv['std']*100:.2f}%",             GOLD),
    ("Vocab Size",       f"{tfidf_s['vocab_size']:,}",         TEAL),
    ("Sparsity",         f"{tfidf_s['sparsity']*100:.1f}%",   '#8b949e'),
    ("Train Samples",    f"{tfidf_s['train_shape'][0]}",       '#8b949e'),
    ("Test Samples",     f"{tfidf_s['test_shape'][0]}",        '#8b949e'),
]
for i, (lbl, val, col) in enumerate(stat_items):
    y = 0.92 - i * 0.115
    ax6.text(0.05, y, lbl, transform=ax6.transAxes, fontsize=10, color='#8b949e')
    ax6.text(0.95, y, val, transform=ax6.transAxes, fontsize=11, color=col,
             ha='right', fontweight='bold')

# 7. Top positive features
ax7 = fig.add_subplot(gs[2, :2])
top_pos = top_f['positive'][:12]
top_neg = top_f['negative'][:12]
words = [w for w,_ in top_pos[::-1]] + [w for w,_ in top_neg]
coefs = [c for _,c in top_pos[::-1]] + [c for _,c in top_neg]
colors_feat = [GREEN if c > 0 else RED for c in coefs]
y_pos = range(len(words))
ax7.barh(list(y_pos), coefs, color=colors_feat, edgecolor='none', alpha=0.85)
ax7.set_yticks(list(y_pos)); ax7.set_yticklabels(words, fontsize=9)
ax7.axvline(0, color='#8b949e', linewidth=1)
ax7.set_title('Top Predictive Features (Logistic Regression Coefficients)',
              fontsize=12, color='#f0f6fc', pad=8)
ax7.set_xlabel('Coefficient Value')
ax7.set_facecolor(CARD); ax7.spines['top'].set_visible(False); ax7.spines['right'].set_visible(False)

# 8. CV scores
ax8 = fig.add_subplot(gs[2, 2])
fold_labels = [f'Fold {i+1}' for i in range(len(cv['scores']))]
ax8.bar(fold_labels, cv['scores']*100, color=PURP, edgecolor='none', alpha=0.85)
ax8.axhline(cv['mean']*100, color=GOLD, linestyle='--', linewidth=2,
            label=f"Mean = {cv['mean']*100:.1f}%")
ax8.set_title('5-Fold Cross-Validation', fontsize=12, color='#f0f6fc', pad=8)
ax8.set_ylabel('Accuracy (%)'); ax8.set_ylim(60, 105)
ax8.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
for i, (lbl, val) in enumerate(zip(fold_labels, cv['scores'])):
    ax8.text(i, val*100+0.5, f"{val*100:.1f}%", ha='center', fontsize=9, color='#e6edf3')
ax8.set_facecolor(CARD); ax8.spines['top'].set_visible(False); ax8.spines['right'].set_visible(False)

plt.savefig('outputs/sentiment_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✅ Dashboard saved → outputs/sentiment_dashboard.png")

# ══ Figure 2: Text Preprocessing Pipeline ═════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig2.suptitle('Text Preprocessing & TF-IDF Analysis',
              fontsize=15, color='#f0f6fc', fontweight='bold', y=1.02)

# Preprocessing word count change
ax_a = axes2[0]
df['clean_wc'] = df['review'].apply(lambda t: len(sa.preprocess(t).split()))
ax_a.scatter(df[df.sentiment=='positive']['word_count'],
             df[df.sentiment=='positive']['clean_wc'],
             color=GREEN, alpha=0.6, s=40, label='Positive')
ax_a.scatter(df[df.sentiment=='negative']['word_count'],
             df[df.sentiment=='negative']['clean_wc'],
             color=RED, alpha=0.6, s=40, label='Negative')
ax_a.plot([0,60],[0,60], color='#8b949e', linestyle='--', linewidth=1, alpha=0.5)
ax_a.set_title('Word Count: Raw vs Cleaned', fontsize=12, color='#f0f6fc', pad=8)
ax_a.set_xlabel('Original'); ax_a.set_ylabel('After Preprocessing')
ax_a.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor='#e6edf3')
ax_a.set_facecolor(CARD); ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)

# TF-IDF feature importance heatmap (top 10 × first 20 reviews)
ax_b = axes2[1]
X_te_tfidf = sa.tfidf.transform(sa.X_test[:20])
coef = sa.model.coef_[0]
top_idx = np.argsort(np.abs(coef))[-10:]
feat_matrix = X_te_tfidf[:, top_idx].toarray().T
top_feat_names = [sa.feature_names[i] for i in top_idx]
im = ax_b.imshow(feat_matrix, cmap='Blues', aspect='auto')
ax_b.set_yticks(range(10)); ax_b.set_yticklabels(top_feat_names, fontsize=9)
ax_b.set_xlabel('Review (test set)'); ax_b.set_title('TF-IDF Matrix (Top 10 Features)',
               fontsize=12, color='#f0f6fc', pad=8)
plt.colorbar(im, ax=ax_b, shrink=0.8)
ax_b.set_facecolor(CARD)

# Prediction confidence histogram
ax_c = axes2[2]
correct_prob   = sa.y_prob[sa.y_pred == sa.y_test]
incorrect_prob = sa.y_prob[sa.y_pred != sa.y_test]
ax_c.hist(correct_prob, bins=12, color=GREEN, alpha=0.7, label='Correct', edgecolor='none')
if len(incorrect_prob):
    ax_c.hist(incorrect_prob, bins=6, color=RED, alpha=0.7, label='Incorrect', edgecolor='none')
ax_c.set_title('Prediction Confidence', fontsize=12, color='#f0f6fc', pad=8)
ax_c.set_xlabel('P(Positive)'); ax_c.set_ylabel('Count')
ax_c.legend(facecolor=PANEL, edgecolor='#30363d', labelcolor='#e6edf3')
ax_c.set_facecolor(CARD); ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

for ax in axes2:
    ax.set_facecolor(CARD)
fig2.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig('outputs/preprocessing_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✅ Preprocessing analysis saved → outputs/preprocessing_analysis.png")

# ── Text report ───────────────────────────────────────────────────────────────
lines = [
    "=" * 65,
    "  SENTIMENT ANALYSIS SYSTEM — FULL REPORT",
    "  IMDB + Amazon Reviews | TF-IDF | Logistic Regression",
    "=" * 65,
    "",
    "── 1. DATASET OVERVIEW ──────────────────────────────────────────",
    f"  Total Reviews    : {len(df)}",
    f"  IMDB Reviews     : {len(df[df.source=='IMDB'])}",
    f"  Amazon Reviews   : {len(df[df.source=='Amazon'])}",
    f"  Positive         : {df.label.sum()}",
    f"  Negative         : {(df.label==0).sum()}",
    f"  Avg Word Count   : {df['word_count'].mean():.1f} words",
    "",
    "── 2. TEXT PREPROCESSING ────────────────────────────────────────",
    "  Steps applied:",
    "  • Lowercase conversion",
    "  • HTML tag removal  (regex)",
    "  • URL / email removal",
    "  • Special character stripping (keep apostrophes)",
    "  • Stopword removal  (NLTK English, negations preserved)",
    "  • Porter Stemming   (e.g. 'running' → 'run')",
    f"  Avg tokens before : {df['word_count'].mean():.1f}",
    f"  Avg tokens after  : {df['clean_wc'].mean():.1f}",
    f"  Reduction         : {(1 - df['clean_wc'].mean()/df['word_count'].mean())*100:.1f}%",
    "",
    "── 3. TF-IDF VECTORIZATION ──────────────────────────────────────",
    f"  Vocabulary Size   : {tfidf_s['vocab_size']:,} features",
    f"  N-gram Range      : (1, 2) — unigrams + bigrams",
    f"  Train Matrix      : {tfidf_s['train_shape'][0]} × {tfidf_s['train_shape'][1]}",
    f"  Test Matrix       : {tfidf_s['test_shape'][0]} × {tfidf_s['test_shape'][1]}",
    f"  Sparsity          : {tfidf_s['sparsity']*100:.1f}%",
    f"  Avg Non-zero Feat : {tfidf_s['avg_nonzero_feats']} per document",
    "  Settings: sublinear_tf=True, min_df=1, max_df=0.95",
    "",
    "── 4. LOGISTIC REGRESSION ───────────────────────────────────────",
    "  solver=lbfgs, C=1.0, class_weight=balanced, max_iter=1000",
    "",
    "── 5. MODEL PERFORMANCE ─────────────────────────────────────────",
    f"  Accuracy          : {metrics['accuracy']*100:.2f}%",
    f"  AUC-ROC           : {metrics['auc_roc']:.4f}",
    "",
    f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Support':>9}",
    "  " + "─" * 52,
]
for cls in ['Negative', 'Positive']:
    r = rep[cls]
    lines.append(
        f"  {cls:<12} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1-score']:>10.3f} {int(r['support']):>9}"
    )
lines += [
    f"  {'Macro Avg':<12} {rep['macro avg']['precision']:>10.3f} {rep['macro avg']['recall']:>8.3f} {rep['macro avg']['f1-score']:>10.3f}",
    "",
    "── 6. CROSS-VALIDATION (5-FOLD STRATIFIED) ──────────────────────",
    f"  Scores  : {[f'{s*100:.1f}%' for s in cv['scores']]}",
    f"  Mean    : {cv['mean']*100:.2f}%",
    f"  Std Dev : ±{cv['std']*100:.2f}%",
    "",
    "── 7. TOP PREDICTIVE FEATURES ───────────────────────────────────",
    "",
    f"  {'POSITIVE FEATURES':<35} {'NEGATIVE FEATURES'}",
    "  " + "─" * 65,
]
for (pw, pc), (nw, nc) in zip(top_f['positive'][:12], top_f['negative'][:12]):
    lines.append(f"  {pw:<30} {pc:+.4f}    {nw:<30} {nc:+.4f}")

lines += [
    "",
    "── 8. LIVE INFERENCE RESULTS ────────────────────────────────────",
    f"  {'Review (truncated)':<52} {'Sent':<12} {'Conf':>6}",
    "  " + "─" * 72,
]
for p in predictions:
    sent = "POSITIVE" if p['sentiment'] == 'positive' else "NEGATIVE"
    lines.append(f"  {p['text']:<52} {sent:<12} {p['confidence']*100:>5.1f}%")

lines += [
    "",
    "── 9. CONFUSION MATRIX ──────────────────────────────────────────",
    f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}",
    "",
    "── 10. METHODOLOGY ──────────────────────────────────────────────",
    "  Pipeline: Raw text → Preprocess → TF-IDF → Logistic Regression",
    "  Libraries: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn",
    "",
    "=" * 65,
    "  END OF REPORT",
    "=" * 65,
]

report = "\n".join(lines)
print("\n" + report)
with open('outputs/sentiment_report.txt', 'w') as f:
    f.write(report)

print("\n✅ All outputs generated:")
print("   • outputs/sentiment_dashboard.png")
print("   • outputs/preprocessing_analysis.png")
print("   • outputs/sentiment_report.txt")
