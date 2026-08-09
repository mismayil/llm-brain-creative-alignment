"""
Layer-wise correlation profiles.

QUESTION
--------
Two observations in the paper both key on token pooling:
  (a) peak-alignment depth is early under last-token pooling and late under
      mean-token pooling;
  (b) the size correlation appears under last-token pooling, the AUT-score
      correlation under mean-token pooling.
Is (b) a consequence of (a) -- i.e. is DEPTH the operative variable -- or are
both just separate consequences of pooling, with depth incidental?

TEST
----
Instead of correlating model attributes against the BEST-LAYER alignment, we
correlate them against alignment at EVERY layer, and plot the correlation as a
function of relative depth. Two curves per pooling:
    r( size,      alignment(d) )  vs  d
    r( aut_score, alignment(d) )  vs  d

  - If depth is operative: WITHIN a pooling, the size curve should peak early
    and the AUT-score curve should peak late.
  - If pooling is operative and depth incidental: within a pooling each curve
    is roughly flat, and the curves differ BETWEEN poolings instead.

This also sidesteps the best-layer optimism bias entirely, since we read fixed
layers rather than maxima.

BINNING
-------
Models have different layer counts (16-80 here), so we bin by RELATIVE depth.
Each model contributes, per bin, the median over its layers falling in that bin
and over subjects. Correlations are then computed across models within each bin.

CAVEATS (report these)
----------------------
  - n ~= 23-24 models per bin, so individual bins are noisy; read the shape of
    the curve, not single bins. Bootstrap bands are shown.
  - Binning smooths over genuine layer-to-layer structure.
  - size and aut_score use different prompt variants ('main' and 'scoring'),
    so the two curves are not computed on identical alignment values. Within
    each curve the comparison across depth IS internally consistent, which is
    what the depth hypothesis requires.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from dotenv import load_dotenv

load_dotenv()  # for CADABRA_DATA_DIR

from cadabra.utils import read_json
from cadabra.alignment.alignment_utils import get_alignment_runs

# ---------------------------------------------------------------------------
RESULTS_JSON = Path("experiments/templeton_aut/data/main_results.json")
OUTDIR = Path("experiments/templeton_aut/data")

NETWORK_CONTAINS = "yeo_dmn"
STAGE = "prompt"
N_BINS = 10
N_BOOT = 2_000
RNG = np.random.default_rng(0)

# attribute -> prompt variant, matching the main analyses
ATTR_VARIANT = {"size_B": "main", "aut_score": "scoring"}
# ---------------------------------------------------------------------------


def entry_key(e):
    """(pooling, variant) for a matched AUT-AUT entry in the target cell, or None."""
    bn, ds = e["brain_network"], e["dataset"]
    if NETWORK_CONTAINS not in bn:
        return None
    if "create" not in bn or "create" not in ds:      # brain=AUT and model=AUT
        return None
    if e["activation_mode"] != STAGE or e["nc_threshold"] != 0:
        return None
    pooling = "mean_token" if "mean" in e["model_data_sampling"] else "last_token"
    variant = "scoring" if "short_eval" in ds else "main"
    return pooling, variant


def fetch_scores(entry):
    """Return (n_layers, n_subjects) array of noise-ceiling-adjusted scores.

    Chain: get_alignment_runs -> run.config['output_path'] -> metadata json ->
    data.noise_ceiling_adjusted_scores_path (relative to output_path.parent).
    """
    dataset_parts = entry["dataset"].split("(")
    runs = get_alignment_runs(
        alignment_method="rsa_per_subject",
        nc_threshold=0.0,
        brain_network=entry["brain_network"],
        dataset=f"{dataset_parts[0]}{entry['model_name']}_.*",
        model_data_sampling=entry["model_data_sampling"],
        model_network=entry["model_network"],
        activation_mode=entry["activation_mode"]
    )
    runs = list(runs)
    if not runs:
        raise RuntimeError(f"no run found for {entry['model_name']}")
    if len(runs) > 1:
        print(f"    warning: {len(runs)} runs matched for "
              f"{entry['model_name']}, using the first")

    out_path = Path(runs[0].config["output_path"])
    meta = read_json(str(out_path))
    rel = meta["data"]["noise_ceiling_adjusted_scores_path"]
    arr = np.load(out_path.parent / rel)
    arr = np.asarray(arr, float)
    if arr.ndim == 3:                 # (num_tokens=1, num_layers, num_subjects)
        assert arr.shape[0] == 1, f"unexpected num_tokens={arr.shape[0]}"
        arr = arr[0]
    return arr                        # (num_layers, num_subjects)


def binned_profile(A, n_bins=N_BINS):
    """Median-over-subjects alignment per layer, averaged into relative-depth bins.
    Returns an array of length n_bins (NaN where a bin has no layers)."""
    per_layer = np.nanmedian(A, axis=1)                  # (n_layers,)
    n_layers = len(per_layer)
    depth = np.arange(n_layers) / max(n_layers - 1, 1)   # 0 .. 1
    edges = np.linspace(0, 1, n_bins + 1)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (depth >= lo) & (depth <= hi if b == n_bins - 1 else depth < hi)
        if sel.any():
            out[b] = np.nanmedian(per_layer[sel])
    return out


def corr_with_ci(x, y, n_boot=N_BOOT, rng=RNG):
    """Pearson r plus percentile bootstrap CI over models."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    r, p = pearsonr(x, y)
    boot = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        boot.append(pearsonr(x[idx], y[idx])[0])
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return float(r), float(p), float(lo), float(hi), n


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    entries = read_json(str(RESULTS_JSON))

    # group entries by (pooling, variant)
    groups = {}
    for e in entries:
        k = entry_key(e)
        if k:
            groups.setdefault(k, []).append(e)
    print("cells found:", {k: len(v) for k, v in groups.items()})

    rows = []
    for (pooling, variant), es in sorted(groups.items()):
        attrs = [a for a, v in ATTR_VARIANT.items() if v == variant]
        if not attrs:
            continue
        profiles, meta = [], []
        for e in es:
            try:
                A = fetch_scores(e)
            except Exception as exc:
                print(f"  skip {e['model_name']} ({pooling},{variant}): {exc}")
                continue
            profiles.append(binned_profile(A))
            meta.append({"model": e["model_name"], "size_B": e["model_size_b"],
                         "aut_score": e.get("aut_score", np.nan),
                         "n_layers": A.shape[0]})

        if len(profiles) < 4:
            print(f"  too few models for ({pooling},{variant})")
            continue
        P = np.vstack(profiles)                      # (n_models, n_bins)
        M = pd.DataFrame(meta)
        centres = (np.linspace(0, 1, N_BINS + 1)[:-1]
                   + np.linspace(0, 1, N_BINS + 1)[1:]) / 2

        for attr in attrs:
            xv = M[attr].to_numpy(float)
            for b in range(N_BINS):
                r, p, lo, hi, n = corr_with_ci(xv, P[:, b])
                rows.append({"pooling": pooling, "attribute": attr,
                             "bin": b, "depth": centres[b], "n": n,
                             "r": r, "p": p, "ci_lo": lo, "ci_hi": hi})

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "layerwise_correlation_profiles.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n" + "=" * 88)
    print("LAYER-WISE CORRELATION PROFILES  (DMN, prompt stage)")
    print("  r between model attribute and alignment, per relative-depth bin")
    print("=" * 88)
    for (pool, attr), g in df.groupby(["pooling", "attribute"]):
        print(f"\n  {pool} | {attr}   (n={int(g['n'].iloc[0])} models)")
        print("    depth:  " + "  ".join(f"{d:5.2f}" for d in g["depth"]))
        print("    r    :  " + "  ".join(f"{v:+5.2f}" for v in g["r"]))
        print("    p    :  " + "  ".join(f"{v:5.3f}" for v in g["p"]))
        peak = g.loc[g["r"].idxmax()]
        print(f"    -> max r = {peak['r']:+.3f} at depth {peak['depth']:.2f} "
              f"(p={peak['p']:.4f})")

    # ---- figure: one panel per pooling, one line per attribute ----
    pools = sorted(df["pooling"].unique())
    fig, axes = plt.subplots(1, len(pools), figsize=(5.2 * len(pools), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    colours = {"size_B": "#2f5c8a", "aut_score": "#c0632f"}
    labels = {"size_B": "model size", "aut_score": "AUT score"}
    for ax, pool in zip(axes, pools):
        for attr, g in df[df.pooling == pool].groupby("attribute"):
            g = g.sort_values("depth")
            ax.plot(g["depth"], g["r"], marker="o", ms=4,
                    color=colours.get(attr), label=labels.get(attr, attr))
            ax.fill_between(g["depth"], g["ci_lo"], g["ci_hi"],
                            color=colours.get(attr), alpha=0.15, linewidth=0)
        ax.axhline(0, ls="--", lw=1, color="grey")
        ax.set_title(f"pooling: {pool}")
        ax.set_xlabel("relative layer depth")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("correlation with alignment")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Correlation with model attributes across layer depth (DMN, prompt)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "layerwise_correlation_profiles.png", dpi=200)
    plt.close(fig)

    print(f"\n  wrote {OUTDIR / 'layerwise_correlation_profiles.csv'}")
    print(f"  wrote {OUTDIR / 'layerwise_correlation_profiles.png'}")


if __name__ == "__main__":
    main()