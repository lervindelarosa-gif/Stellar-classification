"""Model building, evaluation and tuning for the Stellar Classification project."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

GRADIENT_BOOSTING_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [200, 400],
    "max_leaf_nodes": [31, 63],
}


def get_models():
    """Return the baseline and the four models to compare, untrained.

    Logistic Regression and KNN measure distances between feature values, so they get a
    StandardScaler inside a pipeline (fitted on the training data only, so no leakage).
    """
    return {
        "Baseline (most common)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Fit each model and return a table of test accuracy, macro precision, macro recall and macro F1.

    "Macro" means the score is computed for each class, then averaged, so every class counts equally.
    zero_division=0 scores a class the model never predicts (e.g. the baseline) as 0 instead of warning.
    """
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"model": name,
                     "accuracy": accuracy_score(y_test, pred),
                     "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
                     "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
                     "macro_F1": f1_score(y_test, pred, average="macro", zero_division=0)})
    return pd.DataFrame(rows).set_index("model").round(4)


def tune_gradient_boosting(X_train, y_train, param_grid=GRADIENT_BOOSTING_GRID):
    """Grid search with 5-fold stratified cross-validation, scored on macro F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                        param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid
