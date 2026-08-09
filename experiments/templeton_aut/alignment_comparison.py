"""
Paired comparison of matched-task alignment: AUT (brain=AUT, model=AUT) vs
OCT (brain=OCT, model=OCT), across models.

Motivation: the correlations with model size / AUT score are present in the AUT
condition and absent in the OCT condition, but the OCT *alignment values
themselves* are substantively high. This script asks the separate question of
whether alignment is on average higher in the AUT condition, and whether any
such difference is systematic across models or driven by a few outliers.

Because the same models are evaluated in both conditions, the comparison is
PAIRED: one AUT-minus-OCT difference per model, tested against zero with
Wilcoxon signed-rank (no normality assumption at n~24) plus a sign test as a
distribution-free cross-check. Effect size is the median paired difference with
a bootstrap CI.

CAVEAT (state in the paper): alignment values here are noise-ceiling normalized,
and the ceiling is estimated separately for the AUT and OCT brain data. If the
two conditions differ in neural reliability, part of any difference (or lack of
one) reflects the ceiling rather than representational similarity. This script
operates on normalized values only and cannot separate those contributions.

INPUT: main_results.json, same schema used by correlation_robustness.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

RESULTS_JSON = Path("experiments/templeton_aut/data/main_results.json")
OUTDIR = Path("experiments/templeton_aut/data")

RNG = np.random.default_rng(0)
N_BOOT = 10_000
CI = 95.0

# Figure is drawn for this cell; the table covers all network x stage x pooling.
FIG_NETWORK = "DMN"
FIG_STAGE = "prompt"
FIG_POOLING = "last_token"

brain_label_mapping = {
    "yeo_dmn.*dt_create_with_ratings.json": "DMN",
    "yeo_fp.*dt_create_with_ratings.json": "FPN",
    "yeo_som.*dt_create.json": "SOM",
    "yeo_dmn.*dt_object.json": "DMN",
    "yeo_fp.*dt_object.json": "FPN",
}


def load_data() -> pd.DataFrame:
    """Long-format frame, matching correlation_robustness.py's loader."""
    from cadabra.utils import read_json

    results = read_json(str(RESULTS_JSON))
    rows = []
    for r in results:
        if r["nc_threshold"] != 0:
            continue
        rows.append({
            "model": r["model_name"],
            "network": brain_label_mapping.get(r["brain_network"], r["brain_network"]),
            "brain_task": "AUT" if "create" in r["brain_network"] else "OCT",
            "model_task": "AUT" if "create" in r["dataset"] else "OCT",
            "stage": r["activation_mode"],
            "pooling": "mean_token" if "mean" in r["model_data_sampling"] else "last_token",
            "alignment": r["alignment"],
            "prompt_variant": "scoring" if "short_eval" in r["dataset"] else "main",
        })
    return pd.DataFrame(rows)


def paired_frame(df, network, stage, pooling):
    """Return one row per model with matched AUT and OCT alignment.

    Only the 'main' prompt variant is used: the 'scoring' variant exists for the
    AUT condition only, so including it would break the pairing."""
    d = df[(df.network == network) & (df.stage == stage)
           & (df.pooling == pooling) & (df.prompt_variant == "main")]

    aut = d[(d.brain_task == "AUT") & (d.model_task == "AUT")][["model", "alignment"]]
    oct_ = d[(d.brain_task == "OCT") & (d.model_task == "OCT")][["model", "alignment"]]
    m = aut.merge(oct_, on="model", suffixes=("_a", "_b"))
    m["diff"] = m["alignment_a"] - m["alignment_b"]

    only_a = set(aut.model) - set(oct_.model)
    only_b = set(oct_.model) - set(aut.model)
    return m, sorted(only_a), sorted(only_b)


def paired_frame_networks(df, net_a, net_b, stage, pooling):
    """Return one row per model with alignment in two NETWORKS, holding the task
    fixed at the matched AUT condition (brain=AUT, model=AUT).

    Used for the DMN vs SOM comparison: SOM is the control network, so a
    higher DMN alignment is the expected direction."""
    d = df[(df.stage == stage) & (df.pooling == pooling)
           & (df.prompt_variant == "main")
           & (df.brain_task == "AUT") & (df.model_task == "AUT")]

    a = d[d.network == net_a][["model", "alignment"]]
    b = d[d.network == net_b][["model", "alignment"]]
    m = a.merge(b, on="model", suffixes=("_a", "_b"))
    m["diff"] = m["alignment_a"] - m["alignment_b"]

    only_a = set(a.model) - set(b.model)
    only_b = set(b.model) - set(a.model)
    return m, sorted(only_a), sorted(only_b)


def bootstrap_median_ci(v, n_boot=N_BOOT, ci=CI):
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return float("nan"), float("nan")
    boot = np.array([np.median(v[RNG.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


def compare(m):
    """Paired stats on the (condition A minus condition B) differences."""
    d = m["diff"].to_numpy()
    n = len(d)
    nz = d[d != 0]
    w_p = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) else 1.0
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    s_p = (binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided").pvalue
           if (n_pos + n_neg) else 1.0)
    ci_lo, ci_hi = bootstrap_median_ci(d)
    return {
        "n": n,
        "median_a": float(np.median(m["alignment_a"])),
        "median_b": float(np.median(m["alignment_b"])),
        "median_diff": float(np.median(d)),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_a_higher": n_pos, "n_b_higher": n_neg,
        "p_wilcoxon": w_p, "p_sign": float(s_p),
    }


def plot_paired(m, label_a, label_b, title, outpath):
    """Alignment in condition A vs condition B per model, with identity line.
    Points above the line are models where A exceeds B."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(m["alignment_b"], m["alignment_a"], s=30, alpha=0.7,
               edgecolor="black", linewidth=0.4)

    lo = float(min(m["alignment_b"].min(), m["alignment_a"].min()))
    hi = float(max(m["alignment_b"].max(), m["alignment_a"].max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1.0,
            color="grey", zorder=0, label="equal alignment")

    ax.set_xlabel(f"{label_b} alignment")
    ax.set_ylabel(f"{label_a} alignment")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


def to_markdown(t: pd.DataFrame) -> str:
    head = "| " + " | ".join(t.columns) + " |"
    sep = "|" + "|".join(["---"] * len(t.columns)) + "|"
    lines = [head, sep]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    networks = [n for n in ("DMN", "FPN") if n in set(df.network)]
    stages = sorted(df.stage.unique())
    poolings = sorted(df.pooling.unique())

    # ---------------- comparison 1: task (AUT vs OCT), per network -----------
    print("=" * 100)
    print("(1) PAIRED AUT vs OCT ALIGNMENT  (matched task conditions, per model)")
    print("  diff = alignment(brain=AUT, model=AUT) - alignment(brain=OCT, model=OCT)")
    print("  Wilcoxon signed-rank + sign test, two-sided; CI is bootstrap on median diff.")
    print("=" * 100)

    rows = []
    for net in networks:
        for stage in stages:
            for pooling in poolings:
                m, only_a, only_b = paired_frame(df, net, stage, pooling)
                if len(m) < 4:
                    print(f"  {net} | {stage} | {pooling}: only {len(m)} paired "
                          f"models, skipped")
                    continue
                if only_a or only_b:
                    print(f"  NOTE {net} | {stage} | {pooling}: unpaired models "
                          f"dropped -- AUT-only={only_a}, OCT-only={only_b}")
                rows.append({"comparison": "AUT vs OCT", "network": net,
                             "stage": stage, "pooling": pooling, **compare(m)})

    # ---------------- comparison 2: network (DMN vs SOM), AUT task -----------
    print("\n" + "=" * 100)
    print("(2) PAIRED DMN vs SOM ALIGNMENT  (AUT task, per model)")
    print("  diff = alignment(DMN) - alignment(SOM), both at brain=AUT, model=AUT")
    print("  SOM is the control network: a positive difference is the expected direction.")
    print("=" * 100)

    if "SOM" in set(df.network):
        for stage in stages:
            for pooling in poolings:
                m, only_a, only_b = paired_frame_networks(
                    df, "DMN", "SOM", stage, pooling)
                if len(m) < 4:
                    print(f"  DMN vs SOM | {stage} | {pooling}: only {len(m)} "
                          f"paired models, skipped")
                    continue
                if only_a or only_b:
                    print(f"  NOTE DMN vs SOM | {stage} | {pooling}: unpaired "
                          f"models dropped -- DMN-only={only_a}, SOM-only={only_b}")
                rows.append({"comparison": "DMN vs SOM", "network": "DMN/SOM",
                             "stage": stage, "pooling": pooling, **compare(m)})
    else:
        print("  SOM not present in the loaded data, skipped.")

    # ---------------- output --------------------------------------------------
    tbl = pd.DataFrame(rows)
    show = tbl[["comparison", "network", "stage", "pooling", "n",
                "median_a", "median_b", "median_diff", "ci_lo", "ci_hi",
                "n_a_higher", "n_b_higher", "p_wilcoxon", "p_sign"]]
    md = to_markdown(show)
    print("\n" + md)
    (OUTDIR / "alignment_condition_comparisons.md").write_text(md)
    tbl.to_csv(OUTDIR / "alignment_condition_comparisons.csv", index=False)

    # ---------------- figures for the primary cell ---------------------------
    m, _, _ = paired_frame(df, FIG_NETWORK, FIG_STAGE, FIG_POOLING)
    if len(m) >= 4:
        plot_paired(m, f"AUT ({FIG_NETWORK})", f"OCT ({FIG_NETWORK})",
                    f"{FIG_NETWORK} | {FIG_STAGE} | pooling: {FIG_POOLING}",
                    OUTDIR / "aut_vs_oct_alignment.png")

    if "SOM" in set(df.network):
        m2, _, _ = paired_frame_networks(df, "DMN", "SOM", FIG_STAGE, FIG_POOLING)
        if len(m2) >= 4:
            plot_paired(m2, "DMN", "SOM",
                        f"AUT | {FIG_STAGE} | pooling: {FIG_POOLING}",
                        OUTDIR / "dmn_vs_som_alignment.png")

    print(f"  wrote {OUTDIR / 'alignment_condition_comparisons.md'}")


if __name__ == "__main__":
    main()