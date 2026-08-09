"""
Bootstrap error bars + difference-of-differences significance tests
for the high- vs low-creativity post-training analysis (Figure 4).

All resampling is OVER SUBJECTS, because subjects are the unit of
replication: each subject contributes one alignment value per
(model, population) cell, and the reported point estimate is the
median across subjects (matching Section 3.3).

INPUT FORMAT
------------
A long-format table `df` with columns:
    subject     : subject id (str/int)
    model       : model variant, e.g. 'crpo-llama-3.1-8b-instruct-cre'
    population  : 'high' or 'low'
    alignment   : per-subject RSA alignment value (signed, noise-ceiling
                  normalized -- whatever Figure 4's y-axis actually is)

If your data is stored differently (nested dicts, per-model .npy, etc.),
replace ONLY the `load_data()` function. Nothing downstream changes.
"""

import numpy as np
import pandas as pd
from cadabra.utils import read_json

RNG = np.random.default_rng(0)      # fixed seed -> reproducible CIs/p-values
N_BOOT = 10_000                      # bootstrap resamples
N_PERM = 10_000                      # permutation iterations
STAT = np.median                     # point estimate; matches "median across subjects"

# ----------------------------------------------------------------------
# 0. LOAD DATA  -- replace this stub with your real loader
# ----------------------------------------------------------------------
def load_data(model_sampling="time:mean::layer:1:", 
              activation_mode="prompt",
              nc_threshold=0.0,
              rating_threshold=2) -> pd.DataFrame:
    """Return long-format df: subject, model, population, alignment.

    >>> REPLACE THIS BODY with code that loads your cached per-subject
    >>> alignment values. The synthetic block below only exists so the
    >>> script runs end-to-end and you can sanity-check the logic.
    """

    results = read_json(f"experiments/templeton_aut/data/high_low_cre_results_{rating_threshold}.json")

    models_to_compare = [
        "llama-3.1-8b-instruct",
        "llama-3.1-8b",
        "llama-3.1-minitaur-8b",
        "crpo-llama-3.1-8b-instruct-cre",
        "deepseek-r1-distill-llama-8b",
        "deepseek-r1-distill-qwen-7b",
        "qwen2.5-math-7b",
        "qwen2.5-7b",
        "qwen2.5-7b-instruct",
        "crpo-sft-llama-3.1-8b-instruct",
        "crpo-dpo-llama-3.1-8b-instruct"
    ]

    data = []

    for result in results:
        if result["model_name"] in models_to_compare and result["nc_threshold"] == nc_threshold and result["activation_mode"] == activation_mode and result["model_data_sampling"] == model_sampling:
            for subj, hi_align, lo_align in zip(result["high_run_subjects"], result["high_run_best_layer_alignments"], result["low_run_best_layer_alignments"]):
                data.append({
                    "subject": subj,
                    "model": result["model_name"],
                    "population": "high",
                    "alignment": hi_align
                })
                data.append({
                    "subject": subj,
                    "model": result["model_name"],
                    "population": "low",
                    "alignment": lo_align
                })
    return pd.DataFrame(data, columns=["subject", "model", "population", "alignment"])


# ----------------------------------------------------------------------
# 1. Reshape to a subjects x (model, population) matrix
# ----------------------------------------------------------------------
def to_matrix(df: pd.DataFrame):
    """Pivot to wide: index=subject, columns=(model, population).
 
    Returns the pivoted DataFrame plus the array of subject ids, so that
    bootstrap resampling can index whole subjects (all their cells move
    together -- this preserves within-subject correlation across models
    and populations, which is what makes the paired contrasts valid).
    """
    wide = df.pivot_table(
        index="subject", columns=["model", "population"], values="alignment"
    )
    return wide, wide.index.to_numpy()
 
 
# ----------------------------------------------------------------------
# 2. ERROR BARS: bootstrap CI on each bar of Figure 4
# ----------------------------------------------------------------------
def bootstrap_bars(wide, n_boot=N_BOOT, ci=95.0):
    """Bootstrap (over subjects) the median alignment for every
    (model, population) cell. Returns a tidy df with point estimate
    and CI bounds -- one row per bar in Figure 4.
    """
    cols = wide.columns                       # MultiIndex (model, population)
    vals = wide.to_numpy()                    # shape (n_subj, n_cells)
    n_subj = vals.shape[0]
 
    boot = np.empty((n_boot, vals.shape[1]))
    for b in range(n_boot):
        idx = RNG.integers(0, n_subj, n_subj)         # resample subjects
        boot[b] = STAT(vals[idx], axis=0)              # ignores NaN? no -> see note
    # If any cell can contain NaN (subject missing a model), use nan-aware:
    #   boot[b] = np.nanmedian(vals[idx], axis=0)
 
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    out = pd.DataFrame({
        "model": [c[0] for c in cols],
        "population": [c[1] for c in cols],
        "point": STAT(vals, axis=0),
        "ci_lo": np.percentile(boot, lo_q, axis=0),
        "ci_hi": np.percentile(boot, hi_q, axis=0),
    })
    return out
 
 
# ----------------------------------------------------------------------
# 3. SIGNED PER-ARM ALIGNMENT  (the core analysis)
#    The claim is about the SIGN of each arm separately, not the gap:
#        align(high) vs 0   and   align(low) vs 0
#    A model's "type" is the PAIR of signs:
#        CrPO       : high +, low -   (selective for high creativity)
#        R1-Distill : high -, low +   (the reversal)
#        Instruct   : high +, low +   (aligned with both)
#    The gap high-low is deliberately NOT the headline -- it collapses
#    e.g. (+0.08, -0.01) into a single small number and discards the very
#    sign information we want to report.
#
#    TWO complementary tests per arm, because the per-subject medians have
#    very wide CIs / heavy tails (see real data):
#      (A) signed median vs 0  -- "is the typical subject's alignment
#          significantly nonzero, and what sign?"  (bootstrap + sign-flip)
#      (B) subject sign-consistency vs 50% -- "do subjects AGREE on the
#          sign?"  (sign / binomial test, robust to heavy tails)
# ----------------------------------------------------------------------
 
def arm_signed_median(values, n_boot=N_BOOT, n_perm=N_PERM, ci=95.0):
    """Test (A): is STAT(values) significantly != 0, and what sign?
 
    values : per-subject alignment for ONE (model, population) arm.
    bootstrap CI + two-sided bootstrap p against 0, plus a sign-flip
    permutation p (flip the sign of each subject's value independently --
    the natural null for 'alignment is symmetric about 0')."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    n = len(v)
    point = STAT(v)
 
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        boot[b] = STAT(v[idx])
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_lo, ci_hi = np.percentile(boot, [lo_q, hi_q])
    p_boot = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
 
    # sign-flip permutation: null = distribution symmetric about 0
    null = np.empty(n_perm)
    for k in range(n_perm):
        signs = RNG.choice([-1.0, 1.0], size=n)
        null[k] = STAT(v * signs)
    p_perm = (np.abs(null) >= np.abs(point)).mean()
 
    if ci_lo > 0:
        sign = "+"
    elif ci_hi < 0:
        sign = "-"
    else:
        sign = "0"   # CI straddles zero -> sign not established
    return {"point": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_boot": p_boot, "p_perm": p_perm, "sign": sign}
 
 
def arm_sign_consistency(values):
    """Test (B): do subjects agree on the sign of alignment?
 
    Exact two-sided sign test of #(value>0) against Binomial(n, 0.5),
    ignoring exact zeros. Returns the fraction positive, the p-value, and
    the established sign ('+' if majority positive & p<.05, '-' if majority
    negative & p<.05, else '0'). Robust to the heavy tails that blow up the
    median CI, because it only uses the sign of each subject's value."""
    from scipy.stats import binomtest
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    nonzero = v[v != 0]
    n = len(nonzero)
    n_pos = int((nonzero > 0).sum())
    frac_pos = n_pos / n if n else np.nan
    p = binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue if n else 1.0
    if p < 0.05 and frac_pos > 0.5:
        sign = "+"
    elif p < 0.05 and frac_pos < 0.5:
        sign = "-"
    else:
        sign = "0"
    return {"frac_pos": frac_pos, "n": n, "p": p, "sign": sign}
 
 
def classify_pattern(sign_high, sign_low):
    """Map a (high, low) sign pair to a human-readable model type.
    Uses only signs that were actually established ('+'/'-'); '0' means
    'not established', so any pattern touching a 0 is reported as partial."""
    pat = (sign_high, sign_low)
    table = {
        ("+", "-"): "selective for HIGH creativity (high +, low -)",
        ("-", "+"): "REVERSAL: aligned with LOW, anti high (high -, low +)",
        ("+", "+"): "aligned with BOTH (high +, low +)",
        ("-", "-"): "anti BOTH (high -, low -)",
    }
    if pat in table:
        return table[pat]
    return f"partial / not fully established (high {sign_high}, low {sign_low})"
 
 
# ----------------------------------------------------------------------
# 5. Multiple-comparison correction across the contrasts you report
# ----------------------------------------------------------------------
def holm_bonferroni(pvals_dict):
    """Holm-Bonferroni correction. Input: {label: p}. Returns
    {label: (p_raw, p_adj, reject@0.05)}. Holm is uniformly more powerful
    than plain Bonferroni and needs no independence assumption."""
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max = {}, 0.0
    for i, (label, p) in enumerate(items):
        p_adj = min((m - i) * p, 1.0)
        running_max = max(running_max, p_adj)   # enforce monotonicity
        out[label] = (p, running_max, running_max < 0.05)
    return out
 
 
# ----------------------------------------------------------------------
# 6. BETWEEN-MODEL CONTRAST ON A SINGLE ARM  (rank/sign based)
#    For ONE population, is model A aligned differently from model B?
#    Paired over subjects. Because alignment is heavily CLIPPED at +/-1
#    (~44% of values), the median of the paired differences is destroyed
#    by exact ties (both models pinned at the same bound -> diff 0), so a
#    median-of-differences test spuriously returns 0 / p=1. We therefore
#    use rank- and sign-based paired tests, which are the correct tools
#    for paired, non-normal, tie-heavy data:
#       - Wilcoxon signed-rank (primary): uses signed ranks of the paired
#         differences; standard paired nonparametric test.
#       - Paired sign test (backup): counts subjects with A>B vs A<B,
#         drops ties, binomial vs 50%; ignores magnitude entirely so the
#         +/-1 clipping is irrelevant.
#    Effect size reported as the median paired difference AND the fraction
#    of subjects favoring A, since the rank statistic itself isn't a "diff".
#    Two-sided.
# ----------------------------------------------------------------------
def arm_contrast(wide, model_a, model_b, population):
    """Paired two-sided Wilcoxon signed-rank + sign test of A vs B on one arm."""
    from scipy.stats import wilcoxon, binomtest
    a = wide[(model_a, population)].to_numpy()
    b = wide[(model_b, population)].to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    d = a - b
    n = len(d)
 
    # --- Wilcoxon signed-rank (primary) ---
    nz = d[d != 0]                       # Wilcoxon drops zero-differences
    n_nonzero = len(nz)
    if n_nonzero > 0:
        # zero_method='wilcox' already drops zeros; pass nonzero explicitly
        w_stat, p_wilcox = wilcoxon(nz, alternative="two-sided")
    else:
        w_stat, p_wilcox = np.nan, 1.0
 
    # --- paired sign test (backup) ---
    n_pos = int((d > 0).sum())           # subjects where A > B
    n_neg = int((d < 0).sum())           # subjects where A < B
    n_used = n_pos + n_neg               # ties dropped
    p_sign = binomtest(n_pos, n_used, 0.5,
                       alternative="two-sided").pvalue if n_used else 1.0
    frac_favor_a = n_pos / n_used if n_used else np.nan
 
    return {
        "median_diff": float(np.median(d)),
        "n": n, "n_nonzero": n_nonzero, "n_ties": n - n_nonzero,
        "p_wilcox": float(p_wilcox),
        "n_pos": n_pos, "n_neg": n_neg, "frac_favor_a": frac_favor_a,
        "p_sign": float(p_sign),
    }
 
 
# ----------------------------------------------------------------------
# 7. TARGETED TESTS for the three reviewer-response claims
# ----------------------------------------------------------------------
def arm_vs_zero_onesided(values, direction, n_boot=N_BOOT, n_perm=N_PERM, ci=95.0):
    """One-sided test that STAT(values) is on the predicted side of 0.
    direction=-1 tests '< 0', direction=+1 tests '> 0'. Robust to the
    +/-1 clipping because it is sign/rank based, not magnitude based:
      - sign-flip permutation p (null: symmetric about 0)
      - sign test p (subjects on predicted side vs 50%)
    Reports a one-sided bootstrap confidence bound on the relevant side."""
    assert direction in (-1, +1)
    from scipy.stats import binomtest
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    n = len(v); point = STAT(v)
 
    boot = np.empty(n_boot)
    for k in range(n_boot):
        boot[k] = STAT(v[RNG.integers(0, n, n)])
    if direction == -1:
        bound = np.percentile(boot, ci)          # one-sided UPPER bound
        bound_lbl = f"one-sided {ci:.0f}% upper bound"
    else:
        bound = np.percentile(boot, 100 - ci)    # one-sided LOWER bound
        bound_lbl = f"one-sided {ci:.0f}% lower bound"
 
    # sign-flip permutation, one-sided in predicted direction
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = STAT(v * RNG.choice([-1.0, 1.0], size=n))
    if direction == -1:
        p_perm = (null <= point).mean()
    else:
        p_perm = (null >= point).mean()
 
    # one-sided sign test on subject signs
    nz = v[v != 0]; n_nz = len(nz)
    n_on_side = int((nz < 0).sum()) if direction == -1 else int((nz > 0).sum())
    p_sign = binomtest(n_on_side, n_nz, 0.5, alternative="greater").pvalue \
        if n_nz else 1.0
    frac_on_side = n_on_side / n_nz if n_nz else np.nan
 
    return {"point": point, "bound": bound, "bound_lbl": bound_lbl,
            "p_perm": p_perm, "p_sign": p_sign,
            "frac_on_side": frac_on_side, "n_nz": n_nz, "n": n}
 
 
def gap_vs_zero_twosided(wide, model, n_perm=N_PERM):
    """Two-sided test that the within-model high-low gap differs from 0,
    via Wilcoxon signed-rank on the per-subject paired differences
    (high - low). Used for the 'no significant difference' (claim 3)
    framing: a NON-significant result means we fail to find a high-vs-low
    difference for this model. (Does not prove equivalence.)"""
    from scipy.stats import wilcoxon, binomtest
    hi = wide[(model, "high")].to_numpy()
    lo = wide[(model, "low")].to_numpy()
    mask = ~(np.isnan(hi) | np.isnan(lo))
    d = hi[mask] - lo[mask]
    n = len(d)
    nz = d[d != 0]; n_nz = len(nz)
    if n_nz > 0:
        w, p_wilcox = wilcoxon(nz, alternative="two-sided")
    else:
        w, p_wilcox = np.nan, 1.0
    n_pos = int((d > 0).sum()); n_neg = int((d < 0).sum())
    p_sign = binomtest(n_pos, n_pos + n_neg, 0.5,
                       alternative="two-sided").pvalue if (n_pos + n_neg) else 1.0
    return {"median_gap": float(np.median(d)), "n": n,
            "n_ties": n - n_nz, "p_wilcox": float(p_wilcox),
            "n_pos": n_pos, "n_neg": n_neg, "p_sign": float(p_sign)}
 
def arm_tost_zero(values, margin, n_boot=N_BOOT, ci=90.0):
    """TOST equivalence test: is STAT(values) practically equivalent to 0,
    i.e. inside the band [-margin, +margin]?
 
    Equivalence is concluded iff the value is significantly GREATER than
    -margin AND significantly LESS than +margin (two one-sided tests).
    Operationally: a (1-2*alpha) CI lying entirely within (-margin, +margin)
    => equivalent at level alpha. With alpha=0.05 that is a 90% CI.
 
    Bootstrap over subjects (sign/rank-robust to the +/-1 clipping). Returns
    the two one-sided p-values, their max (the TOST p), the 90% CI, and the
    verdict. NOTE: equivalence is only as meaningful as `margin`; state and
    justify it. A non-equivalent result does NOT mean 'non-zero'."""
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    n = len(v); point = STAT(v)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        boot[k] = STAT(v[RNG.integers(0, n, n)])
    # one-sided p that value > -margin  (lower equivalence bound)
    p_lower = (boot <= -margin).mean()
    # one-sided p that value < +margin  (upper equivalence bound)
    p_upper = (boot >= +margin).mean()
    p_tost = max(p_lower, p_upper)            # both must be small
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_lo, ci_hi = np.percentile(boot, [lo_q, hi_q])
    equivalent = (ci_lo > -margin) and (ci_hi < margin)
    return {"point": point, "margin": margin, "ci": ci,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
            "equivalent": equivalent}
 
 
def arm_vs_zero_both_onesided(values, n_boot=N_BOOT, n_perm=N_PERM, ci=95.0):
    """For ONE (model, arm): report BOTH one-sided tests against 0 --
    p(>0) and p(<0) -- via sign-flip permutation AND sign test. Sign/rank
    robust to the +/-1 clipping.
 
    NOTE on interpretation: reporting both directions is descriptive. A
    one-sided claim is only valid in the direction predicted a-priori for
    that model/arm; do not pick 'whichever side is significant' post-hoc
    (that inflates the false-positive rate to a two-sided 2*alpha)."""
    from scipy.stats import binomtest
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    n = len(v); point = STAT(v)
 
    boot = np.empty(n_boot)
    for k in range(n_boot):
        boot[k] = STAT(v[RNG.integers(0, n, n)])
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_lo, ci_hi = np.percentile(boot, [lo_q, hi_q])
 
    # sign-flip permutation null, both one-sided p's
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = STAT(v * RNG.choice([-1.0, 1.0], size=n))
    p_perm_gt = (null >= point).mean()   # one-sided p that value > 0
    p_perm_lt = (null <= point).mean()   # one-sided p that value < 0
 
    # one-sided sign tests
    nz = v[v != 0]; n_nz = len(nz); n_pos = int((nz > 0).sum())
    p_sign_gt = binomtest(n_pos, n_nz, 0.5, alternative="greater").pvalue \
        if n_nz else 1.0
    p_sign_lt = binomtest(n_pos, n_nz, 0.5, alternative="less").pvalue \
        if n_nz else 1.0
    frac_pos = n_pos / n_nz if n_nz else np.nan
 
    # conservative per-direction significance (both tests must agree)
    sig_pos = max(p_perm_gt, p_sign_gt) < 0.05
    sig_neg = max(p_perm_lt, p_sign_lt) < 0.05
    return {"point": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "frac_pos": frac_pos, "n": n, "n_nz": n_nz,
            "p_perm_gt": p_perm_gt, "p_sign_gt": p_sign_gt, "sig_pos": sig_pos,
            "p_perm_lt": p_perm_lt, "p_sign_lt": p_sign_lt, "sig_neg": sig_neg}

def compute_stats(df):
    wide, _ = to_matrix(df)
    models = list(wide.columns.get_level_values(0).unique())
 
    print("=" * 104)
    print("PER-MODEL x PER-ARM ALIGNMENT vs 0   (BOTH one-sided directions; sign/rank robust)")
    print("  Each cell reports the one-sided test that alignment is > 0 AND that it is < 0.")
    print("  p_perm = sign-flip permutation   p_sign = one-sided sign test")
    print("  sig+/sig- require BOTH tests < 0.05 in that direction.")
    print("  Claim only the direction your hypothesis predicts a-priori (see note in code).")
    print("=" * 104)
 
    # print(f"  {'model':<32s} {'pop':>4s} {'median':>7s} {'95% CI':>18s} {'%+':>5s} | "
    #       f"{'p>0 perm':>8s} {'p>0 sign':>8s} {'>0':>3s} | "
    #       f"{'p<0 perm':>8s} {'p<0 sign':>8s} {'<0':>3s}")
    
    outputs = []

    outputs.append(f"  {'model':<32s} {'pop':>4s} {'median':>7s} {'95% CI':>14s} {'    sig. alignment':>10s}")
    outputs.append("  " + "-" * 120)
 
    table = []

    for m in models:
        for pop in ("high", "low"):
            r = arm_vs_zero_both_onesided(wide[(m, pop)].to_numpy())
            tagp = " + " if r["sig_pos"] else "   "
            tagn = " - " if r["sig_neg"] else "   "
            ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
            # print(f"  {m:<32s} {pop:>4s} {r['point']:+7.3f} {ci:>18s} "
            #       f"{100*r['frac_pos']:>4.0f}% | "
            #       f"{r['p_perm_gt']:>8.4f} {r['p_sign_gt']:>8.4f} {tagp} | "
            #       f"{r['p_perm_lt']:>8.4f} {r['p_sign_lt']:>8.4f} {tagn}")
            alignment = "+/-"
            if -0.1 < r['ci_lo'] < r['ci_hi'] <= 1.0:   
                alignment = "+"
            elif -1.0 <= r['ci_lo'] < r['ci_hi'] < 0.1:
                alignment = "-"
            outputs.append(f"  {m:<32s} {pop:>4s} {r['point']:+7.3f} {ci:>18s} "
                  f"{alignment:>10s}")
            table.append({"model": m, "arm": pop, "median": r["point"],
                          "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                          "frac_pos": r["frac_pos"],
                          "p_gt_perm": r["p_perm_gt"], "p_gt_sign": r["p_sign_gt"],
                          "sig_positive": r["sig_pos"],
                          "p_lt_perm": r["p_perm_lt"], "p_lt_sign": r["p_sign_lt"],
                          "sig_negative": r["sig_neg"]})
 
    stats = pd.DataFrame(table)
    return stats, outputs

# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    activation_modes = ["prompt", "model_resp"]
    model_samplings = ["time:mean::layer:1:", "time:-1::layer:1:"]
    nc_threshold = 0
    rating_thresholds = [1.75, 2, 2.25]

    for activation_mode in activation_modes:
        for model_sampling in model_samplings:
            for rating_threshold in rating_thresholds:
                df = load_data(activation_mode=activation_mode, model_sampling=model_sampling, 
                               nc_threshold=nc_threshold, rating_threshold=rating_threshold)
                # df.to_csv(f"per_subject_alignments_{activation_mode}_{model_sampling}_nc{nc_threshold}.csv", index=False)
                stats, outputs = compute_stats(df)
                stats.to_csv(f"experiments/templeton_aut/data/hl_alignment_stats_{activation_mode}_{model_sampling}_nc{nc_threshold}_r{rating_threshold}.csv", index=False)

                print("\n".join(outputs))

                with open(f"experiments/templeton_aut/data/hl_alignment_stats_{activation_mode}_{model_sampling}_nc{nc_threshold}_r{rating_threshold}.txt", "w") as f:
                    f.write("\n".join(outputs))