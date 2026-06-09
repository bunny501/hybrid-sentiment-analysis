
from sklearn.model_selection import GridSearchCV

PARAM_GRID = {
    "tfidf__max_features":[3000,5000,10000],
    "tfidf__ngram_range":[(1,1),(1,2)],
    "clf__C":[0.01,0.1,1,10]
}
