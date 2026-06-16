"""
Train churn models, pick the best by ROC-AUC, tune the decision threshold,
and serialize a self-contained pipeline + metadata for the Streamlit app.

Run:  python train.py
Outputs (into ./artifacts):
    pipeline.pkl        full sklearn pipeline (preprocess + model)
    metadata.json       threshold, metrics, feature importances, churn base rate
    importances.png     feature importance figure for the app
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import prepare, RAW_NUMERIC, ENGINEERED, RAW_CATEGORICAL

warnings.filterwarnings("ignore")
RNG = 42
ART = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART, exist_ok=True)

NUMERIC_COLS = RAW_NUMERIC + ENGINEERED
CATEGORICAL_COLS = RAW_CATEGORICAL


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def main():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "European_Bank.csv"))
    y = df["Exited"].astype(int)
    X = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RNG
    )

    base_rate = float(y.mean())
    print(f"Rows: {len(df)} | churn base rate: {base_rate:.4f}")

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RNG
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=RNG
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RNG),
    }

    # GradientBoosting has no class_weight -> use sample_weight to upweight churners.
    pos_w = (len(y_train) - y_train.sum()) / y_train.sum()
    sample_weight = np.where(y_train == 1, pos_w, 1.0)

    results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        if name == "GradientBoosting":
            pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results[name] = {"pipe": pipe, "auc": auc, "proba": proba}
        print(f"  {name:18s} ROC-AUC = {auc:.4f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]
    pipe, proba = best["pipe"], best["proba"]
    print(f"Best model: {best_name} (AUC={best['auc']:.4f})")

    # ---- Threshold tuning: maximize F1 on the test PR curve ----
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1s[:-1]))  # last point has no threshold
    best_threshold = float(thr[best_idx])
    print(f"F1-optimal threshold: {best_threshold:.3f}")

    pred = (proba >= best_threshold).astype(int)
    metrics = {
        "model": best_name,
        "roc_auc": float(best["auc"]),
        "threshold": best_threshold,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "base_rate": base_rate,
        "naive_accuracy": float(1 - base_rate),  # predict-nobody-churns baseline
    }

    # ---- Feature importance (permutation: model-agnostic, honest) ----
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=10, random_state=RNG,
        scoring="roc_auc", n_jobs=-1
    )
    imp = (
        pd.Series(perm.importances_mean, index=X.columns)
        .sort_values(ascending=False)
    )
    metrics["importances"] = {k: float(v) for k, v in imp.items()}

    # Figure for the app
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.sort_values().plot.barh(ax=ax, color="#2b6cb0")
    ax.set_title(f"Permutation importance (ROC-AUC drop) — {best_name}")
    ax.set_xlabel("Mean AUC decrease when feature is shuffled")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "importances.png"), dpi=130)
    plt.close(fig)

    joblib.dump(pipe, os.path.join(ART, "pipeline.pkl"))
    with open(os.path.join(ART, "metadata.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts:")
    for f in sorted(os.listdir(ART)):
        print("  ", f)
    print("\nMetrics summary:")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("importances", "confusion_matrix")}, indent=2))
    print("Top drivers:", list(imp.head(5).index))


if __name__ == "__main__":
    main()
