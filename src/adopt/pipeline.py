"""Descriptive analysis, competing-risks curves, adoption model and figures. Run: python -m adopt.pipeline"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import data, model, style, survival

OUT = Path("reports")
COLORS = dict(zip(data.GROUP_ORDER, [style.SERIES[2], style.SERIES[0], style.SERIES[3], style.SERIES[7]], strict=True))


def fig_outcomes_by_year(df: pd.DataFrame, figs: Path) -> pd.DataFrame:
    closed = df[~df["censored"]]
    share = closed.groupby("year")["outcome"].value_counts(normalize=True).unstack()[data.GROUP_ORDER]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for g in data.GROUP_ORDER:
        ax.plot(share.index, share[g] * 100, marker="o", ms=4, color=COLORS[g])
        ax.text(share.index[-1] + 0.15, share[g].iloc[-1] * 100, g, color=COLORS[g], va="center", fontsize=9)
    ax.set_xlim(share.index[0] - 0.3, share.index[-1] + 2.2)
    ax.set_xticks(share.index)
    ax.set_ylabel("% of dogs leaving the shelter")
    ax.set_title("Adoption tripled while transfers to rescues halved")
    ax.set_ylim(0, None)
    fig.savefig(figs / "outcomes_by_year.png")
    plt.close(fig)
    return share


def fig_cumulative_incidence(df: pd.DataFrame, figs: Path) -> pd.DataFrame:
    cif = survival.cumulative_incidence(df["days"], df["outcome"], horizon=120)
    naive = survival.naive_km_adoption(df["days"], df["adopted"], horizon=120)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.stackplot(
        cif.index,
        *[cif[g] * 100 for g in data.GROUP_ORDER],
        cif["In shelter"] * 100,
        colors=[COLORS[g] for g in data.GROUP_ORDER] + [style.GRID],
        labels=data.GROUP_ORDER + ["Still in shelter"],
        alpha=0.9,
    )
    ax.plot(naive.index, naive * 100, color=style.INK, ls="--", lw=1.6, label="Adoption, naive Kaplan-Meier")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Days since intake")
    ax.set_ylabel("% of dogs")
    ax.set_title("Where dogs are N days after arriving (competing risks)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    fig.savefig(figs / "cumulative_incidence.png")
    plt.close(fig)
    return cif


def fig_adoption_by_age(df: pd.DataFrame, figs: Path) -> dict:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    at90 = {}
    for i, label in enumerate(data.AGE_LABELS):
        sub = df[df["age_group"] == label]
        cif = survival.cumulative_incidence(sub["days"], sub["outcome"], horizon=120)["Adoption"]
        at90[label] = round(float(cif.loc[90]), 3)
        ax.plot(cif.index, cif * 100, color=style.SERIES[[0, 6, 2, 3, 7][i]], label=f"{label} (n={len(sub):,})")
    ax.set_xlabel("Days since intake")
    ax.set_ylabel("% adopted (cumulative incidence)")
    ax.set_title("Dogs aged 7+ are less than half as likely to be adopted as 1-3-year-olds")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 120)
    fig.savefig(figs / "adoption_by_age.png")
    plt.close(fig)
    return at90


def fig_leakage(res: dict, figs: Path) -> None:
    m = res["models"]
    rows = [
        ("Base rate", m["base_rate"]["roc_auc"], style.MUTED),
        ("Intake-only, 2017-22", m["lightgbm_intake_only"]["roc_auc"], style.SERIES[0]),
        ("Intake-only, 2020-22", m["lightgbm_intake_only_recent_window"]["roc_auc"], style.SERIES[2]),
        ("+ spay/neuter & name\n(recorded at outcome)", m["lightgbm_with_leaky_fields"]["roc_auc"], style.CRITICAL),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color=[r[2] for r in rows][::-1])
    for i, r in enumerate(rows[::-1]):
        ax.text(r[1] + 0.005, i, f"{r[1]:.3f}", va="center", fontsize=9, color=style.INK_2)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("ROC-AUC on dogs arriving in 2023-24 (LightGBM)")
    ax.set_title("Two outcome-time fields inflate the score from 0.67 to 0.87")
    ax.grid(axis="y", visible=False)
    fig.savefig(figs / "leakage_audit.png")
    plt.close(fig)


def fig_calibration(res: dict, figs: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 0.8], [0, 0.8], color=style.MUTED, ls="--", lw=1)
    for key, label, color in (
        ("all_years", "Trained 2017-22", style.SERIES[0]),
        ("recent_window", "Trained 2020-22", style.SERIES[2]),
    ):
        c = res["calibration"][key]
        ax.plot(c["predicted"], c["observed"], marker="o", color=color, label=label)
    ax.set_xlabel("Predicted adoption probability (decile mean)")
    ax.set_ylabel("Observed adoption rate, 2023-24")
    ax.set_title("Calibration under drift")
    ax.legend(loc="lower right")
    fig.savefig(figs / "calibration.png")
    plt.close(fig)


def main() -> None:
    style.apply_style()
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    df = data.load()

    share = fig_outcomes_by_year(df, figs)
    cif = fig_cumulative_incidence(df, figs)
    naive = survival.naive_km_adoption(df["days"], df["adopted"], horizon=180)
    cif180 = survival.cumulative_incidence(df["days"], df["outcome"], horizon=180)
    at90 = fig_adoption_by_age(df, figs)
    res = model.evaluate(df)
    res.pop("_test_pred")
    fig_leakage(res, figs)
    fig_calibration(res, figs)

    adopted = df[df["adopted"]]
    summary = {
        "dogs": len(df),
        "intake_period": [str(df["intake_date"].min().date()), str(df["intake_date"].max().date())],
        "still_in_shelter": int(df["censored"].sum()),
        "outcome_share_by_year": share.round(3).to_dict(orient="index"),
        "cif_day_30": cif.loc[30].round(3).to_dict(),
        "cif_day_180": cif180.loc[180].round(3).to_dict(),
        "naive_km_adoption_day_180": round(float(naive.loc[180]), 3),
        "adoption_cif_day_90_by_age": at90,
        "median_days_to_adoption_by_age": adopted.groupby("age_group", observed=True)["days"].median().to_dict(),
        "adoption_rate_by_intake_type": model.adoption_rate_by(df, "intake_type")["adoption_rate"].round(3).to_dict(),
        "adoption_rate_by_condition": model.adoption_rate_by(df, "intake_condition")["adoption_rate"]
        .round(3)
        .to_dict(),
        "model": res,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    mrows = [{"model": k, **v} for k, v in res["models"].items()]
    lines = [
        "# Results",
        "",
        f"{len(df):,} dog stays, intake {summary['intake_period'][0]} to {summary['intake_period'][1]}.",
        "",
        "## Where dogs are N days after intake (Aalen-Johansen cumulative incidence)",
        "",
        cif180.loc[[7, 30, 90, 180]].mul(100).round(1).to_markdown(),
        "",
        f"Naive Kaplan-Meier for adoption at day 180: {naive.loc[180]:.1%} (vs {cif180.loc[180, 'Adoption']:.1%}).",
        "",
        "## Adoption model, test on 2023-24 intakes",
        "",
        f"Train n={res['n_train']:,} (recent window n={res['n_train_recent']:,}), test n={res['n_test']:,}. "
        f"Adoption base rate: train {res['base_rate_train']:.1%}, test {res['base_rate_test']:.1%}.",
        "",
        pd.DataFrame(mrows).to_markdown(index=False),
        "",
        "Permutation importance (ROC-AUC drop, intake-only LightGBM):",
        "",
        pd.Series(res["importance"], name="auc_drop").to_markdown(),
    ]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
