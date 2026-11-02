"""Competing-risks time-to-outcome: Aalen-Johansen cumulative incidence.

A dog that is returned to its owner on day 3 can no longer be adopted. Treating such stays as
"censored" (a naive Kaplan-Meier for adoption) overstates the chance of adoption, so each outcome is
estimated as a competing risk: CIF_k(t) = sum_{s<=t} S(s-) * d_k(s) / n(s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_incidence(days: pd.Series, outcome: pd.Series, horizon: int = 180) -> pd.DataFrame:
    """Cumulative incidence per outcome for t = 0..horizon days.

    days: time from intake to outcome (or to end of data if still in the shelter)
    outcome: outcome label, or None/NaN for stays still open (censored)
    Returns a DataFrame indexed by day with one column per outcome plus "In shelter".
    """
    t = np.asarray(days, dtype=int)
    ev = pd.Series(outcome).reset_index(drop=True)
    kinds = sorted(ev.dropna().unique())
    n_at_risk = len(t)
    surv = 1.0
    cif = {k: 0.0 for k in kinds}
    rows = []
    for day in range(horizon + 1):
        at_day = t == day
        d_k = {k: int((at_day & ev.eq(k).to_numpy()).sum()) for k in kinds}
        d_all = sum(d_k.values())
        if n_at_risk > 0:
            for k in kinds:
                cif[k] += surv * d_k[k] / n_at_risk
            surv *= 1 - d_all / n_at_risk
        rows.append({"day": day, **cif, "In shelter": surv})
        n_at_risk -= int(at_day.sum())  # events and censored stays leave the risk set
    return pd.DataFrame(rows).set_index("day")


def naive_km_adoption(days: pd.Series, adopted: pd.Series, horizon: int = 180) -> pd.Series:
    """1 - Kaplan-Meier treating every non-adoption outcome as censoring (the biased approach)."""
    t = np.asarray(days, dtype=int)
    a = np.asarray(adopted, dtype=bool)
    n = len(t)
    s = 1.0
    out = []
    for day in range(horizon + 1):
        at_day = t == day
        if n > 0:
            s *= 1 - (at_day & a).sum() / n
        out.append(1 - s)
        n -= int(at_day.sum())
    return pd.Series(out, index=pd.RangeIndex(horizon + 1, name="day"))


def median_days(days: pd.Series, mask: pd.Series) -> float:
    return float(np.median(np.asarray(days)[np.asarray(mask)]))
