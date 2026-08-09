"""
Robustness analyses for the alignment-vs-(size, AUT) correlations reported
across the model x network x task x stage x token-sampling grid.

Answers three reviewer asks:
  (1) Are the n=19 Pearson correlations driven by single outliers?
      -> LEAVE-ONE-OUT jackknife per cell (the primary robustness number)
         + SPEARMAN rank correlation alongside Pearson (diagnoses leverage)
  (2) Are the correlations precise?
      -> 95% BCa BOOTSTRAP CI per cell (secondary, honest about n=19 noise)
  (3) Multiple comparisons across the full grid?
      -> TIERED correction:
            primary family (4 tests, pre-specified) -> Bonferroni
            full grid (all cells)                   -> Benjamini-Hochberg FDR
         Reported side-by-side so the reader sees both.

INPUT
-----
A long-format table from load_data() with columns:
    model, network, task, stage, sampling, alignment, size_B, aut_score
One row per (model, configuration). Attributes (size_B, aut_score) are
per-model; alignment varies across the configuration grid.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, norm

RNG = np.random.default_rng(0)
N_BOOT = 10_000

# ---- pre-specified PRIMARY family of 3 tests ----------------------------
# The Figure 2 + Appendix Figure 6 correlations that the paper makes
# CLAIMS about: alignment ~ {size, AUT} in creativity-relevant networks
# during prompt processing. We pre-specify ONLY the cells the paper claims
# a positive effect for. FPN x size is a PREDICTED null (part of the
# dissociation argument: size matters for DMN but not FPN) and is
# therefore NOT in the corrected family -- it is reported as a predicted
# null alongside the mismatch and Somatomotor controls.
#
# Task structure (4 brain_task x model_task combinations):
#   (AUT,AUT) primary:    matched creative condition -> predict positive
#   (AUT,OCT), (OCT,AUT): MISMATCH controls -> predict null (the real
#                         discriminative test: positive here would mean
#                         the matched effect is generic, not creativity-specific)
#   (OCT,OCT)            : matched non-creative reference -> no clean
#                         prediction (consistent with creativity-specificity
#                         if null, defensible either way)
PRIMARY = [
    # (network, brain_task, model_task, stage, attribute)
    ("DMN", "AUT", "AUT", "prompt", "size_B"),
    ("DMN", "AUT", "AUT", "prompt", "aut_score"),
    ("FPN", "AUT", "AUT", "prompt", "aut_score"),
]

# Per-attribute sampling for the PRIMARY family. Figure 2 uses last_token for
# the size correlation and mean_token for the AUT-score correlation, so this is
# attribute-conditional rather than one global value.
PRIMARY_SAMPLING = {
    "size_B":    "last_token",
    "aut_score": "mean_token",
}


def predicted_outcome(brain_task, model_task, network, attribute):
    """A-priori prediction tag for each cell:
        '+' = primary, predict positive correlation
        '0' = predicted null (the paper's controls)
        '?' = no clean prediction (not part of the paper's design)

    The paper's controls are TASK-MATCHED: the same correlation computed on
    the non-creative OCT task (brain=OCT, model=OCT), and on the Somatomotor
    control network. The cross-task MISMATCH cells (brain != model) are not
    part of the original design -- they arise only from enumerating the full
    grid -- so they carry no a-priori prediction and are reported descriptively.
    """
    # Somatomotor control network -> predicted null regardless of task.
    # NOTE: brain_label_mapping emits "SOM", not "Somatomotor".
    if network in ("SOM", "Somatomotor"):
        return "0"

    # cross-task mismatch: not part of the paper's design
    if brain_task != model_task:
        return "?"

    # matched OCT condition: the paper's non-creative task control
    if brain_task == "OCT":
        return "0"

    # matched AUT condition in a creativity-relevant network
    if network in ("DMN", "FPN"):
        if network == "FPN" and attribute == "size_B":
            return "0"      # documented FPN x size predicted null
        return "+"

    return "?"


# Attribute -> prompt_variant: AUT score correlations use alignment extracted
# from a separate "scoring" generation pass (shorter-response prompt), because
# the scores themselves were computed on those generations. Size correlations
# use the "main" alignment. This mapping is enforced inside the run loop so
# each correlation uses the right alignment column.
ATTRIBUTE_VARIANT = {
    "size_B":    "main",
    "aut_score": "scoring",
}


from cadabra.utils import read_json

brain_label_mapping = {
    "yeo_dmn.*dt_create_with_ratings.json": "DMN",
    "yeo_fp.*dt_create_with_ratings.json": "FPN",
    "yeo_som.*dt_create.json": "SOM",
    "yeo_dmn.*dt_object.json": "DMN",
    "yeo_fp.*dt_object.json": "FPN",
}

# ----------------------------------------------------------------------
# 0. LOAD DATA  -- replace with your real loader.
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Return long-format df with columns:
        model, network, task, stage, sampling, alignment, size_B, aut_score
    One row per (model, configuration). Implement this yourself."""
    results = read_json("experiments/templeton_aut/data/main_results.json")
    df_data = []

    for result in results:
        if result["nc_threshold"] == 0:
            df_data.append({
                "model": result["model_name"],
                "network": brain_label_mapping.get(result["brain_network"], result["brain_network"]),
                "brain_task": "AUT" if "create" in result["brain_network"] else "OCT",
                "model_task": "AUT" if "create" in result["dataset"] else "OCT",
                "stage": result["activation_mode"],
                "sampling": "mean_token" if "mean" in result["model_data_sampling"] else "last_token",
                "alignment": result["alignment"],
                "size_B": result["model_size_b"],
                "aut_score": result["aut_score"],
                "prompt_variant": "scoring" if "short_eval" in result["dataset"] else "main",
            })
    return pd.DataFrame(df_data)




# ----------------------------------------------------------------------
# 1. CORE CORRELATION + ROBUSTNESS BUNDLE PER CELL
# ----------------------------------------------------------------------
def loo_jackknife(x, y, method="pearson"):
    """Drop one (model) at a time, recompute r. Return the n LOO r's plus
    the index of the model whose removal causes the LARGEST change in r
    (the leverage point)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    full_r = (pearsonr if method == "pearson" else spearmanr)(x, y).statistic
    loo = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        loo[i] = (pearsonr if method == "pearson" else spearmanr)(x[m], y[m]).statistic
    leverage_idx = int(np.argmax(np.abs(loo - full_r)))
    return {"full_r": float(full_r),
            "loo_min": float(loo.min()), "loo_max": float(loo.max()),
            "loo_median": float(np.median(loo)),
            "leverage_idx": leverage_idx,
            "leverage_drop_r": float(loo[leverage_idx]),
            "loo_values": loo}


def bca_bootstrap_pearson(x, y, n_boot=N_BOOT, ci=95.0):
    """Bias-corrected and accelerated (BCa) bootstrap CI for Pearson r.
    BCa is the small-n-appropriate variant; plain percentile CIs are
    biased for skewed bootstrap distributions (which r often is)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    theta_hat = pearsonr(x, y).statistic

    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            boot[b] = np.nan        # degenerate resample (all-same x or y)
        else:
            boot[b] = pearsonr(x[idx], y[idx]).statistic
    n_valid = int((~np.isnan(boot)).sum())
    boot = boot[~np.isnan(boot)]
    if n_valid < 0.9 * n_boot:
        # at n=19, more than 10% degenerate resamples means the CI is
        # untrustworthy; flag rather than silently report a misleading number
        return float("nan"), float("nan"), theta_hat

    # bias-correction z0
    prop_lt = (boot < theta_hat).mean()
    prop_lt = np.clip(prop_lt, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop_lt)

    # acceleration a via jackknife
    jack = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        jack[i] = pearsonr(x[m], y[m]).statistic
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    alpha = (100 - ci) / 200
    z_lo, z_hi = norm.ppf(alpha), norm.ppf(1 - alpha)
    a1 = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    return float(np.quantile(boot, a1)), float(np.quantile(boot, a2)), theta_hat


def cell_report(df_cell, attribute):
    """Full robustness bundle for one (network, task, stage, sampling, attribute) cell."""
    d = df_cell.dropna(subset=["alignment", attribute]).copy()
    x, y = d[attribute].to_numpy(), d["alignment"].to_numpy()
    models = d["model"].to_numpy()
    n = len(d)
    if n < 4:
        return {"n": n, "note": "insufficient data"}

    pr = pearsonr(x, y);  sr = spearmanr(x, y)
    p_pearson, p_spearman = float(pr.pvalue), float(sr.pvalue)
    r_pearson, r_spearman = float(pr.statistic), float(sr.statistic)

    loo_p = loo_jackknife(x, y, "pearson")
    loo_s = loo_jackknife(x, y, "spearman")
    ci_lo, ci_hi, _ = bca_bootstrap_pearson(x, y)

    return {
        "n": n,
        "r_pearson": r_pearson, "p_pearson": p_pearson,
        "bca_ci": (ci_lo, ci_hi),
        "r_spearman": r_spearman, "p_spearman": p_spearman,
        "loo_pearson_range": (loo_p["loo_min"], loo_p["loo_max"]),
        "loo_pearson_leverage_model": str(models[loo_p["leverage_idx"]]),
        "loo_pearson_drop_r": loo_p["leverage_drop_r"],
        "loo_spearman_range": (loo_s["loo_min"], loo_s["loo_max"]),
    }


# ----------------------------------------------------------------------
# 2. MULTIPLE-COMPARISON CORRECTION
# ----------------------------------------------------------------------
def bonferroni(pvals_dict, alpha=0.05):
    """Family-wise correction for a SMALL pre-specified family.
    Returns {label: (p_raw, p_adj, reject)}."""
    m = len(pvals_dict)
    out = {}
    for label, p in pvals_dict.items():
        p_adj = min(m * p, 1.0)
        out[label] = (p, p_adj, p_adj < alpha)
    return out


def benjamini_hochberg(pvals_dict, q=0.05):
    """BH-FDR for the FULL grid. Returns {label: (p_raw, q_adj, reject)}.
    Controls expected proportion of false discoveries, much less brutal
    than Bonferroni for large families."""
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    # raw q = p * m / rank, then enforce monotonicity from the largest down
    q_raw = [(label, p, p * m / (i + 1)) for i, (label, p) in enumerate(items)]
    q_adj = [None] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        running_min = min(running_min, q_raw[i][2])
        q_adj[i] = min(running_min, 1.0)
    return {q_raw[i][0]: (q_raw[i][1], q_adj[i], q_adj[i] < q) for i in range(m)}


# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    # ---- enumerate every cell in the grid ----
    # We iterate per (config, attribute) and pick the right prompt_variant
    # for each attribute (size -> main, aut_score -> scoring). Cells where
    # the required variant was not extracted are recorded with a 'missing
    # variant' note rather than silently swapped to the wrong alignment.
    grid_cells = []
    skipped = []
    for (net, btask, mtask, stage, sampling), sub in df.groupby(
            ["network", "brain_task", "model_task", "stage", "sampling"]):
        for attr in ("size_B", "aut_score"):
            variant = ATTRIBUTE_VARIANT[attr]
            sub_v = sub[sub["prompt_variant"] == variant]
            pred = predicted_outcome(btask, mtask, net, attr)
            label = (f"{net} | brain={btask} | model={mtask} | "
                     f"{stage} | {sampling} | {attr} | v={variant}  [{pred}]")
            if len(sub_v) == 0:
                skipped.append((label, f"prompt_variant='{variant}' not extracted"))
                continue
            grid_cells.append((label, net, btask, mtask, stage, sampling,
                               attr, variant, pred, sub_v))

    print("=" * 130)
    print("PER-CELL CORRELATION ROBUSTNESS  (n = number of models per cell)")
    print("  Pearson r, BCa 95% CI, Spearman rho, LOO jackknife range per cell.")
    print("  Headline = Pearson r; Spearman & LOO together diagnose leverage.")
    print("  Prediction tag in [.]:  + primary,  0 predicted null,  ? no clean prediction")
    print("=" * 130)
    hdr = (f"  {'cell':<78s} {'n':>3s} {'r_P':>6s} {'BCa CI':>16s} "
           f"{'r_S':>6s} {'LOO_P range':>16s} {'leverage':>26s}")
    print(hdr); print("  " + "-" * 128)

    full_p = {}                # for BH-FDR over the whole grid
    cell_pred = {}             # label -> prediction tag
    rows = []
    for label, net, btask, mtask, stage, sampling, attr, variant, pred, sub in grid_cells:
        r = cell_report(sub, attr)
        if r.get("note"):
            print(f"  {label:<78s} n={r['n']}  [insufficient]"); continue
        ci = f"[{r['bca_ci'][0]:+.2f},{r['bca_ci'][1]:+.2f}]"
        loo = f"[{r['loo_pearson_range'][0]:+.2f},{r['loo_pearson_range'][1]:+.2f}]"
        print(f"  {label:<78s} {r['n']:>3d} {r['r_pearson']:>+6.2f} {ci:>16s} "
              f"{r['r_spearman']:>+6.2f} {loo:>16s}  "
              f"{r['loo_pearson_leverage_model']:>20s}->{r['loo_pearson_drop_r']:+.2f}")
        full_p[label] = r["p_pearson"]
        cell_pred[label] = pred
        rows.append({"cell": label, "prediction": pred,
                     "prompt_variant": variant,
                     **{k: v for k, v in r.items() if k != "loo_values"}})

    if skipped:
        print("\n  Cells skipped (required prompt_variant not extracted):")
        for label, reason in skipped:
            print(f"    {label}  --  {reason}")

    # ---- primary family: Bonferroni over the pre-specified tests ----
    print("\n" + "=" * 130)
    print(f"PRIMARY FAMILY  (n={len(PRIMARY)} pre-specified tests, brain=AUT, "
            f"model=AUT; sampling per attribute: {PRIMARY_SAMPLING}):  "
            f"Bonferroni-corrected")
    print("=" * 130)
    primary_p = {}
    for (net, btask, mtask, stage, attr) in PRIMARY:
        pred = predicted_outcome(btask, mtask, net, attr)
        variant = ATTRIBUTE_VARIANT[attr]
        sampling = PRIMARY_SAMPLING[attr]          # <- per attribute
        label = (f"{net} | brain={btask} | model={mtask} | "
                f"{stage} | {sampling} | {attr} | v={variant}  [{pred}]")
        if label in full_p:
            primary_p[label] = full_p[label]
        else:
            print(f"  [missing in grid] {label}")
    for label, (p_raw, p_adj, reject) in bonferroni(primary_p).items():
        print(f"  {label:<78s} p={p_raw:.4f}  p_bonf={p_adj:.4f}  "
              f"{'sig @.05' if reject else 'n.s.'}")

    # ---- full grid: BH-FDR ----
    print("\n" + "=" * 130)
    print(f"FULL GRID  (n={len(full_p)} cells):  Benjamini-Hochberg FDR-corrected")
    print("  Cells surviving q<0.05 are discoveries net of expected false-positive rate.")
    print("  Survivor vs prediction agreement:")
    print("    + survives  -> consistent with primary claim")
    print("    0 survives  -> UNEXPECTED null-prediction violation (worth examining)")
    print("    + does NOT survive -> claim should be softened for that cell")
    print("=" * 130)
    bh = benjamini_hochberg(full_p, q=0.05)
    for label, (p_raw, q_adj, reject) in sorted(bh.items(), key=lambda kv: kv[1][0]):
        pred = cell_pred.get(label, "?")
        flag = ""
        if reject and pred == "+":  flag = "<- primary, replicates"
        elif reject and pred == "0": flag = "<- PREDICTED NULL VIOLATED"
        elif reject and pred == "?": flag = "<- unprediction-tagged survivor"
        elif (not reject) and pred == "+": flag = "<- primary does NOT replicate"
        tag = "FDR sig" if reject else "       "
        print(f"  {label:<78s} p={p_raw:.4f}  q={q_adj:.4f}  {tag}  {flag}")

    # ---- summary of dissociation evidence ----
    print("\n" + "=" * 130)
    print("DISSOCIATION SUMMARY  (counts by prediction tag and FDR survival)")
    print("=" * 130)
    from collections import Counter
    survived = Counter((cell_pred[l], "sig" if bh[l][2] else "n.s.") for l in full_p)
    for (pred, status), n in sorted(survived.items()):
        print(f"  prediction={pred}  status={status:>4s}  n={n}")

    # ---- save full per-cell table ----
    out = pd.DataFrame(rows)
    out["bonferroni_primary_p_adj"] = out["cell"].map(
        {lbl: bonferroni(primary_p)[lbl][1] for lbl in primary_p})
    out["bh_fdr_q"] = out["cell"].map({lbl: bh[lbl][1] for lbl in bh})
    out.to_csv("experiments/templeton_aut/data/correlation_robustness_table.csv", index=False)
    print("\n  saved -> experiments/templeton_aut/data/correlation_robustness_table.csv")