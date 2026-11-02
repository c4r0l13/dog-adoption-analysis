"""Will a dog be adopted? A model that only uses what is known on the day it arrives.

Two fields look predictive but are recorded at outcome time, not at intake:
- spay/neuter status inside "sex" (the shelter alters dogs before adoption)
- "animal_name" (strays often receive a name only once they are put up for adoption)
The audit trains the same model with and without them, on a forward-in-time split.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

CATEGORICAL = ["sex_base", "intake_condition", "intake_type", "intake_subtype", "primary_color", "jurisdiction"]
NUMERIC = ["age_years", "month"]
LEAKY = ["altered", "has_name"]
TRAIN_YEARS = range(2017, 2023)
RECENT_YEARS = range(2020, 2023)  # adoption rate doubled after 2019; recent data reflects current practice
TEST_YEARS = range(2023, 2025)


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Closed stays only; rare categories pooled so the encoders see stable levels."""
    d = df[~df["censored"]].copy()
    for c in CATEGORICAL:
        d[c] = d[c].fillna("unknown").astype(str)
        counts = d.loc[d["year"].isin(TRAIN_YEARS), c].value_counts()
        rare = counts[counts < 30].index
        d.loc[d[c].isin(rare) | ~d[c].isin(counts.index), c] = "other"
    d["altered"] = d["altered"].astype(int)
    d["has_name"] = d["has_name"].astype(int)
    return d


def logistic(numeric: list[str], categorical: list[str]):
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=5, degree=3)),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), categorical),
        ]
    )
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))


def gbm(seed: int = 0) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )


def _gbm_matrix(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = d[cols].copy()
    for c in cols:
        if c in CATEGORICAL:
            x[c] = x[c].astype("category")
    return x


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 3),
        "pr_auc": round(float(average_precision_score(y, p)), 3),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def evaluate(df: pd.DataFrame) -> dict:
    d = model_frame(df)
    train, test = d[d["year"].isin(TRAIN_YEARS)], d[d["year"].isin(TEST_YEARS)]
    y_tr, y_te = train["adopted"].to_numpy(), test["adopted"].to_numpy()
    out: dict = {
        "n_train": len(train),
        "n_test": len(test),
        "base_rate_train": round(float(y_tr.mean()), 3),
        "base_rate_test": round(float(y_te.mean()), 3),
        "models": {},
    }

    # Base-rate reference: predict the training adoption rate for everyone.
    out["models"]["base_rate"] = metrics(y_te, np.full(len(y_te), y_tr.mean()))

    for name, extra in (("intake_only", []), ("with_leaky_fields", LEAKY)):
        num = NUMERIC + extra
        lr = logistic(num, CATEGORICAL).fit(train[num + CATEGORICAL], y_tr)
        p_lr = lr.predict_proba(test[num + CATEGORICAL])[:, 1]
        cols = NUMERIC + CATEGORICAL + extra
        gb = gbm().fit(_gbm_matrix(train, cols), y_tr)
        p_gb = gb.predict_proba(_gbm_matrix(test, cols))[:, 1]
        out["models"][f"logistic_{name}"] = metrics(y_te, p_lr)
        out["models"][f"lightgbm_{name}"] = metrics(y_te, p_gb)
        if name == "intake_only":
            out["_test_pred"] = pd.DataFrame({"y": y_te, "p_lr": p_lr, "p_gb": p_gb}, index=test.index)
            pi = permutation_importance(
                gb, _gbm_matrix(test, cols), y_te, scoring="roc_auc", n_repeats=10, random_state=0
            )
            out["importance"] = (
                pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False).round(4).to_dict()
            )

    # Drift: adoption share doubled between 2017-19 and 2022-24. Retrain on the recent window only.
    recent = train[train["year"].isin(RECENT_YEARS)]
    cols = NUMERIC + CATEGORICAL
    gb_recent = gbm().fit(_gbm_matrix(recent, cols), recent["adopted"])
    p_recent = gb_recent.predict_proba(_gbm_matrix(test, cols))[:, 1]
    lr_recent = logistic(NUMERIC, CATEGORICAL).fit(recent[cols], recent["adopted"])
    out["models"]["logistic_intake_only_recent_window"] = metrics(y_te, lr_recent.predict_proba(test[cols])[:, 1])
    out["models"]["lightgbm_intake_only_recent_window"] = metrics(y_te, p_recent)
    out["n_train_recent"] = len(recent)
    out["_test_pred"]["p_recent"] = p_recent

    # Calibration by decile: all-years model vs recent-window model.
    tp = out["_test_pred"]
    out["calibration"] = {}
    for key, col in (("all_years", "p_gb"), ("recent_window", "p_recent")):
        bins = pd.qcut(tp[col], 10, labels=False, duplicates="drop")
        cal = tp.groupby(bins).agg(predicted=(col, "mean"), observed=("y", "mean"))
        out["calibration"][key] = cal.round(3).to_dict(orient="list")
    return out


def adoption_rate_by(df: pd.DataFrame, col: str, min_n: int = 50) -> pd.DataFrame:
    """Share of closed stays ending in adoption by a grouping column (descriptive, not causal)."""
    d = df[~df["censored"]]
    g = d.groupby(col, observed=True)["adopted"].agg(["mean", "size"]).rename(columns={"mean": "adoption_rate"})
    return g[g["size"] >= min_n].sort_values("adoption_rate", ascending=False)
