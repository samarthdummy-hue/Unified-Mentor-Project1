"""
Shared feature engineering.

CRITICAL: this exact function is called in BOTH train.py and app.py.
That is the whole point — training and live inference must transform a row
identically, or the deployed predictions are silently wrong.

Raw input columns expected (one row or many):
    CreditScore, Geography, Gender, Age, Tenure, Balance,
    NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

Dropped before this stage: Year (constant), CustomerId, Surname (identifiers).
"""

import numpy as np
import pandas as pd

# Columns the model is actually trained on (raw + engineered).
RAW_NUMERIC = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
RAW_CATEGORICAL = ["Geography", "Gender"]
ENGINEERED = [
    "BalanceSalaryRatio",
    "ProductDensity",
    "EngagementProduct",
    "AgeTenureInteraction",
    "ZeroBalanceFlag",
]

# Final column order handed to the preprocessing pipeline.
MODEL_INPUT_COLS = RAW_NUMERIC + ENGINEERED + RAW_CATEGORICAL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features the brief asks for. Pure / no fitting.

    Defensive against div-by-zero: Tenure and EstimatedSalary can be 0,
    Balance is 0 for ~36% of customers.
    """
    df = df.copy()

    # Balance-to-salary ratio. +1 guards zero salary; result is a clean ratio.
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)

    # Product density: products held per year of tenure (Tenure can be 0).
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1.0)

    # Engagement x product: an inactive multi-product customer behaves very
    # differently from an active one. Interaction captures that.
    df["EngagementProduct"] = df["IsActiveMember"] * df["NumOfProducts"]

    # Age x tenure: long-tenured older customers are stickier.
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

    # Explicit zero-balance flag (zero balance is a meaningful state here,
    # not missing data — often a fully cross-sold or dormant account).
    df["ZeroBalanceFlag"] = (df["Balance"] == 0).astype(int)

    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, add features, return model-ready columns in order."""
    df = add_features(df)
    return df[MODEL_INPUT_COLS]
