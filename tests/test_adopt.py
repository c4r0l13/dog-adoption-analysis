import numpy as np
import pandas as pd
import pytest

from adopt import data, survival


def raw_rows() -> pd.DataFrame:
    base = {"animal_type": "dog", "animal_name": "Rex", "sex": "Male", "dob": "2020-01-01"}
    rows = [
        {**base, "animal_id": "A1", "intake_date": "2022-01-01", "outcome_date": "2022-01-11",
         "outcome_type": "adoption"},
        {**base, "animal_id": "A2", "intake_date": "2022-01-01", "outcome_date": "2022-01-03",
         "outcome_type": "return to owner", "sex": "Spayed"},
        {**base, "animal_id": "A3", "intake_date": "2022-01-05", "outcome_date": None, "outcome_type": None},
        {**base, "animal_id": "A4", "intake_date": "2022-01-05", "outcome_date": "2022-01-06",
         "outcome_type": "duplicate"},
        {**base, "animal_id": "C1", "animal_type": "cat", "intake_date": "2022-01-01",
         "outcome_date": "2022-01-02", "outcome_type": "adoption"},
    ]  # fmt: skip
    return pd.DataFrame(rows)


def test_clean_filters_and_derives_fields():
    d = data.clean(raw_rows())
    assert list(d["animal_id"]) == ["A1", "A2", "A3"]  # cats and duplicates removed
    assert d.loc[0, "days"] == 10 and d.loc[0, "adopted"]
    assert d.loc[1, "sex_base"] == "female" and d.loc[1, "altered"]
    assert d.loc[2, "censored"] and pd.isna(d.loc[2, "outcome"])
    assert d.loc[2, "days"] == 6  # open stay runs to the last outcome date in the data
    assert d.loc[0, "age_group"] == "1-3 years"


def test_cumulative_incidence_without_censoring_matches_proportions():
    days = pd.Series([1, 2, 2, 3, 5, 5, 8, 10])
    outcome = pd.Series(["A", "B", "A", "A", "B", "A", "B", "A"])
    cif = survival.cumulative_incidence(days, outcome, horizon=12)
    assert cif.loc[12, "A"] == pytest.approx(5 / 8)
    assert cif.loc[12, "B"] == pytest.approx(3 / 8)
    assert cif.loc[2, "A"] == pytest.approx(2 / 8)
    row_sums = cif[["A", "B", "In shelter"]].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0)


def test_competing_risks_vs_naive_km():
    # Half the dogs go home on day 1, the rest are adopted on day 10.
    days = pd.Series([1] * 50 + [10] * 50)
    outcome = pd.Series(["Home"] * 50 + ["Adoption"] * 50)
    cif = survival.cumulative_incidence(days, outcome, horizon=20)
    naive = survival.naive_km_adoption(days, outcome.eq("Adoption"), horizon=20)
    assert cif.loc[20, "Adoption"] == pytest.approx(0.5)
    assert naive.loc[20] == pytest.approx(1.0)  # treating "went home" as censoring claims everyone gets adopted


def test_censored_stays_reduce_risk_set():
    days = pd.Series([2, 4, 4])
    outcome = pd.Series([None, "A", "B"])
    cif = survival.cumulative_incidence(days, outcome, horizon=5)
    assert cif.loc[5, "A"] == pytest.approx(0.5) and cif.loc[5, "B"] == pytest.approx(0.5)
