"""Load the Long Beach shelter records and build one clean row per dog stay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/longbeach.csv")

# Outcome groups used throughout. "Adoption" includes foster-to-adopt placements.
OUTCOME_GROUPS = {
    "adoption": "Adoption",
    "foster to adopt": "Adoption",
    "return to owner": "Returned to owner",
    "rescue": "Rescue / transfer",
    "transfer": "Rescue / transfer",
    "transport": "Rescue / transfer",
    "return to rescue": "Rescue / transfer",
    "homefirst": "Rescue / transfer",
    "euthanasia": "Euthanasia / died",
    "died": "Euthanasia / died",
    "disposal": "Euthanasia / died",
}
GROUP_ORDER = ["Adoption", "Returned to owner", "Rescue / transfer", "Euthanasia / died"]

AGE_BINS = [-np.inf, 0.5, 1, 3, 7, np.inf]
AGE_LABELS = ["< 6 months", "6-12 months", "1-3 years", "3-7 years", "7+ years"]


def age_group(age_years: pd.Series) -> pd.Series:
    return pd.cut(age_years, AGE_BINS, labels=AGE_LABELS, right=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per dog stay with outcome group, length of stay and intake-time attributes.

    Stays with no outcome yet are kept and marked as censored (event = None) at the last date in the data.
    Duplicates and records marked "missing"/"duplicate" are dropped.
    """
    d = raw[raw["animal_type"].eq("dog")].copy()
    for c in ("dob", "intake_date", "outcome_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d = d[~d["outcome_type"].isin(["missing", "duplicate"])]
    d = d.drop_duplicates(subset=["animal_id", "intake_date"])

    end = d["outcome_date"].max()
    d["outcome"] = d["outcome_type"].map(OUTCOME_GROUPS)
    d["censored"] = d["outcome_date"].isna()
    d.loc[d["censored"], "outcome"] = None
    d = d[d["censored"] | d["outcome"].notna()]

    d["days"] = (d["outcome_date"].fillna(end) - d["intake_date"]).dt.days.clip(lower=0)
    age = (d["intake_date"] - d["dob"]).dt.days / 365.25
    d["age_years"] = age.where(age >= 0)
    d["age_group"] = age_group(d["age_years"])
    d["year"] = d["intake_date"].dt.year
    d["month"] = d["intake_date"].dt.month
    # "sex" mixes biology with spay/neuter status, which the shelter updates before adoption.
    d["sex_base"] = d["sex"].map({"Male": "male", "Neutered": "male", "Female": "female", "Spayed": "female"})
    d["sex_base"] = d["sex_base"].fillna("unknown")
    d["altered"] = d["sex"].isin(["Neutered", "Spayed"])
    d["has_name"] = d["animal_name"].notna()
    d["adopted"] = d["outcome"].eq("Adoption")
    return d.reset_index(drop=True)


def load(path: str | Path = RAW) -> pd.DataFrame:
    return clean(pd.read_csv(path))
