"""
Generate a static SHAP summary plot for the trained model.

Run AFTER train.py:  python explain.py
This is a training-time step only. The Streamlit app just displays the saved
PNG, so shap is NOT required at app runtime (keeps the deploy light).
"""

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from features import prepare

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def main():
    pipe = joblib.load(os.path.join(ART, "pipeline.pkl"))
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "European_Bank.csv"))
    X = prepare(df).sample(n=1500, random_state=42)  # subsample for speed

    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    X_enc = prep.transform(X)
    feat_names = prep.get_feature_names_out()
    # Clean up the verbose ColumnTransformer prefixes for readability.
    feat_names = [n.split("__", 1)[-1] for n in feat_names]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_enc)

    plt.figure()
    shap.summary_plot(
        shap_values, X_enc, feature_names=feat_names, show=False, max_display=12
    )
    plt.tight_layout()
    plt.savefig(os.path.join(ART, "shap_summary.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved artifacts/shap_summary.png")


if __name__ == "__main__":
    main()
