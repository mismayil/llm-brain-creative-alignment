"""
Per-subject and per-stimulus sample-size diagnostics for the high vs. low
creativity split (camera-ready appendix).

Produces:
  1. Scatter of per-subject N_low vs N_high (one point per subject, diagonal
     reference line). Points hugging an axis are subjects whose RDM in one
     bucket is estimated from very few responses -- the case that matters for
     the median-across-subjects aggregation.
  2. Sorted stacked bar of per-stimulus counts in each bucket (one bar per
     stimulus, sorted by proportion high).
  3. Summary table of per-subject N (min/median/max) in each bucket, at all
     three cutoffs, written as markdown + CSV.

INPUT: Excel file with columns
    id, condition, stimuli, response, rater1, rater2, rater3, rater4
Rows with condition != 'aut' or missing response are dropped; the final
rating is the mean across the four raters.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
INPUT_XLSX = Path("../data/Templeton_fMRI_data/mri_responses.xlsx")
OUTDIR = Path("experiments/templeton_aut/data")

RATER_COLS = ["rater1", "rater2", "rater3", "rater4"]
CUTOFFS = [1.75, 2.0, 2.25]      # all three reported in the paper
PRIMARY_CUTOFF = 2.0             # cutoff used for the two figures

# If True, a response needs all four ratings to be included. If False, the
# mean is taken over whatever ratings are present. Set deliberately -- it
# changes N. See note printed at runtime.
REQUIRE_ALL_RATERS = False
# ---------------------------------------------------------------------------


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load, filter to AUT, drop missing responses, average the rater columns."""
    df = pd.read_excel(path)

    n_raw = len(df)
    df = df[df["condition"].astype(str).str.strip().str.lower() == "create"].copy()
    n_aut = len(df)

    # drop missing / empty responses
    resp = df["response"].astype(str).str.strip()
    df = df[df["response"].notna() & (resp != "") & (resp.str.lower() != "nan")].copy()
    n_resp = len(df)

    ratings = df[RATER_COLS].apply(pd.to_numeric, errors="coerce")
    n_valid_raters = ratings.notna().sum(axis=1)

    if REQUIRE_ALL_RATERS:
        keep = n_valid_raters == len(RATER_COLS)
    else:
        keep = n_valid_raters > 0
    df, ratings, n_valid_raters = df[keep], ratings[keep], n_valid_raters[keep]

    df["rating"] = ratings.mean(axis=1)
    df["n_raters"] = n_valid_raters.values

    print(f"  rows in file:            {n_raw}")
    print(f"  after condition == aut:  {n_aut}")
    print(f"  after dropping empty:    {n_resp}")
    print(f"  after rating filter:     {len(df)}"
          f"   (REQUIRE_ALL_RATERS={REQUIRE_ALL_RATERS})")
    if not REQUIRE_ALL_RATERS:
        incomplete = int((df["n_raters"] < len(RATER_COLS)).sum())
        if incomplete:
            print(f"  NOTE: {incomplete} responses rated by fewer than "
                  f"{len(RATER_COLS)} raters; mean taken over available ratings.")
    return df


def label_buckets(df: pd.DataFrame, cutoff: float) -> pd.Series:
    """high if rating >= cutoff, else low (matches the paper's convention)."""
    return np.where(df["rating"] >= cutoff, "high", "low")


def per_subject_counts(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """One row per subject: N_low, N_high, total."""
    d = df.assign(bucket=label_buckets(df, cutoff))
    counts = (d.groupby(["id", "bucket"]).size()
                .unstack(fill_value=0)
                .reindex(columns=["low", "high"], fill_value=0))
    counts.columns = ["n_low", "n_high"]
    counts["n_total"] = counts.sum(axis=1)
    return counts.reset_index()


def per_stimulus_counts(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """One row per stimulus: N_low, N_high, total, proportion high."""
    d = df.assign(bucket=label_buckets(df, cutoff))
    counts = (d.groupby(["stimuli", "bucket"]).size()
                .unstack(fill_value=0)
                .reindex(columns=["low", "high"], fill_value=0))
    counts.columns = ["n_low", "n_high"]
    counts["n_total"] = counts.sum(axis=1)
    counts["prop_high"] = counts["n_high"] / counts["n_total"].replace(0, np.nan)
    return counts.reset_index()


def plot_per_subject(subj: pd.DataFrame, cutoff: float, outpath: Path):
    """Scatter of N_low vs N_high, one point per subject, diagonal reference."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(subj["n_low"], subj["n_high"], s=26, alpha=0.65,
               edgecolor="black", linewidth=0.4)

    hi = max(subj["n_low"].max(), subj["n_high"].max())
    ax.plot([0, hi], [0, hi], ls="--", lw=1.0, color="grey", zorder=0,
            label="balanced (N$_{low}$ = N$_{high}$)")

    ax.set_xlabel("N low-creativity responses (per subject)")
    ax.set_ylabel("N high-creativity responses (per subject)")
    ax.set_title(f"Per-subject sample sizes (cutoff = {cutoff})")
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_per_stimulus(stim: pd.DataFrame, cutoff: float, outpath: Path):
    """Stacked bar per stimulus, sorted by proportion high."""
    s = stim.sort_values("prop_high", ascending=False).reset_index(drop=True)
    x = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(max(7.0, 0.19 * len(s)), 4.2))
    ax.bar(x, s["n_low"], label="low creativity", color="#a66b4f")
    ax.bar(x, s["n_high"], bottom=s["n_low"], label="high creativity",
           color="#6b655a")

    ax.set_xticks(x)
    ax.set_xticklabels(s["stimuli"], rotation=90, fontsize=7)
    ax.set_ylabel("N responses")
    ax.set_xlabel("stimulus (sorted by proportion high-creativity)")
    ax.set_title(f"Per-stimulus sample sizes (cutoff = {cutoff})")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


def summary_table(df: pd.DataFrame, cutoffs) -> pd.DataFrame:
    """Per-subject N summary (min/median/max) per bucket, for each cutoff."""
    rows = []
    for c in cutoffs:
        subj = per_subject_counts(df, c)
        d = df.assign(bucket=label_buckets(df, c))
        rows.append({
            "cutoff": c,
            "N_low (total)": int((d["bucket"] == "low").sum()),
            "N_high (total)": int((d["bucket"] == "high").sum()),
            "n_subjects": subj["id"].nunique(),
            "per-subj low: min": int(subj["n_low"].min()),
            "per-subj low: median": float(subj["n_low"].median()),
            "per-subj low: max": int(subj["n_low"].max()),
            "per-subj high: min": int(subj["n_high"].min()),
            "per-subj high: median": float(subj["n_high"].median()),
            "per-subj high: max": int(subj["n_high"].max()),
            "subj with 0 low": int((subj["n_low"] == 0).sum()),
            "subj with 0 high": int((subj["n_high"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def to_markdown(tbl: pd.DataFrame) -> str:
    header = "| " + " | ".join(tbl.columns) + " |"
    sep = "|" + "|".join(["---"] * len(tbl.columns)) + "|"
    lines = [header, sep]
    for _, r in tbl.iterrows():
        cells = [f"{v:g}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Loading and preparing data:")
    df = load_and_prepare(INPUT_XLSX)

    print("\nFigures at primary cutoff:")
    subj = per_subject_counts(df, PRIMARY_CUTOFF)
    stim = per_stimulus_counts(df, PRIMARY_CUTOFF)
    plot_per_subject(subj, PRIMARY_CUTOFF, OUTDIR / "split_per_subject.pdf")
    plot_per_stimulus(stim, PRIMARY_CUTOFF, OUTDIR / "split_per_stimulus.pdf")

    subj.to_csv(OUTDIR / "split_per_subject_counts.csv", index=False)
    stim.to_csv(OUTDIR / "split_per_stimulus_counts.csv", index=False)

    print("\nSummary across cutoffs:")
    tbl = summary_table(df, CUTOFFS)
    md = to_markdown(tbl)
    print(md)
    (OUTDIR / "split_summary.md").write_text(md)
    tbl.to_csv(OUTDIR / "split_summary.csv", index=False)
    print(f"  wrote {OUTDIR / 'split_summary.md'}")


if __name__ == "__main__":
    main()