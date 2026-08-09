"""
Read Gemini AUT scoring JSONs per model, parse the score strings to floats,
and print a markdown table of statistics.

Two aggregation strategies are reported:
  - global:   flatten all items, compute stats once. Simple. Weights by item count.
  - per-item: median across generations within each item (controls within-item
              noise), then mean/median/std across items. Matches how human AUT
              scores are typically aggregated, and is the primary number for
              cross-model comparisons.

PATHS DICT: paste GEMINI_AUT_SCORING_PATHS into the dict below or import it.
ID FIELD:   defaults to 'id'. If your items use a different field, set ID_FIELD.
            If multiple generations per stimulus share an id, per-item aggregation
            will pool them as intended. If each generation has a unique id with
            no stimulus grouping, per-item collapses to global.
"""

import json
import random
import re
import statistics as st
from pathlib import Path

# ---- PASTE YOUR PATHS HERE -------------------------------------------------
GEMINI_AUT_SCORING_PATHS = {
    "llama-3.1-8b-instruct": "../outputs/templeton_aut/aut/llama-3.1-8b-instruct/20260306_140725/aut_scoring/templeton_aut_create_short_eval_data_llama-3.1-8b-instruct_20260306_140725_aut_scoring_gemini-3-flash-preview_20260322_191202.json",
    "crpo-llama-3.1-8b-instruct-cre": "../outputs/templeton_aut/aut/crpo-llama-3.1-8b-instruct-cre/20260306_140819/aut_scoring/templeton_aut_create_short_eval_data_crpo-llama-3.1-8b-instruct-cre_20260306_140819_aut_scoring_gemini-3-flash-preview_20260322_191307.json",
    "gemma-3-270m-it": "../outputs/templeton_aut/aut/gemma-3-270m-it/20260316_153058/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-270m-it_20260316_153058_aut_scoring_gemini-3-flash-preview_20260322_191429.json",
    "gemma-3-1b-it": "../outputs/templeton_aut/aut/gemma-3-1b-it/20260316_153639/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-1b-it_20260316_153639_aut_scoring_gemini-3-flash-preview_20260322_191634.json",
    "gemma-3-4b-it": "../outputs/templeton_aut/aut/gemma-3-4b-it/20260316_154445/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-4b-it_20260316_154445_aut_scoring_gemini-3-flash-preview_20260322_191843.json",
    "gemma-3-12b-it": "../outputs/templeton_aut/aut/gemma-3-12b-it/20260316_155546/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-12b-it_20260316_155546_aut_scoring_gemini-3-flash-preview_20260322_192034.json",
    "gemma-3-27b-it": "../outputs/templeton_aut/aut/gemma-3-27b-it/20260316_161156/aut_scoring/templeton_aut_create_short_eval_data_gemma-3-27b-it_20260316_161156_aut_scoring_gemini-3-flash-preview_20260322_192142.json",
    "llama-3.2-1b-instruct": "../outputs/templeton_aut/aut/llama-3.2-1b-instruct/20260316_163439/aut_scoring/templeton_aut_create_short_eval_data_llama-3.2-1b-instruct_20260316_163439_aut_scoring_gemini-3-flash-preview_20260322_192245.json",
    "llama-3.2-3b-instruct": "../outputs/templeton_aut/aut/llama-3.2-3b-instruct/20260316_163502/aut_scoring/templeton_aut_create_short_eval_data_llama-3.2-3b-instruct_20260316_163502_aut_scoring_gemini-3-flash-preview_20260322_192432.json",
    "olmo-3.1-32b-instruct": "../outputs/templeton_aut/aut/olmo-3.1-32b-instruct/20260322_145141/aut_scoring/templeton_aut_create_short_eval_data_olmo-3.1-32b-instruct_20260322_145141_aut_scoring_gemini-3-flash-preview_20260322_192601.json",
    "llama-3.1-70b-instruct": "../outputs/templeton_aut/aut/llama-3.1-70b-instruct/20260322_145227/aut_scoring/templeton_aut_create_short_eval_data_llama-3.1-70b-instruct_20260322_145227_aut_scoring_gemini-3-flash-preview_20260322_192725.json",
    "llama-3.1-8b": "../outputs/templeton_aut/aut/llama-3.1-8b/20260327_200542/templeton_aut_create_short_eval_data_llama-3.1-8b_20260327_200542.json",
    "llama-3.1-minitaur-8b": "../outputs/templeton_aut/aut/llama-3.1-minitaur-8b/20260327_201250/templeton_aut_create_short_eval_data_llama-3.1-minitaur-8b_20260327_201250.json",
    "qwen2.5-32b-instruct": "../outputs/templeton_aut/aut/qwen2.5-32b-instruct/20260328_210352/templeton_aut_create_short_eval_data_qwen2.5-32b-instruct_20260328_210352.json",
    "qwen2.5-72b-instruct": "../outputs/templeton_aut/aut/qwen2.5-72b-instruct/20260328_210430/templeton_aut_create_short_eval_data_qwen2.5-72b-instruct_20260328_210430.json",
    "deepseek-r1-distill-llama-8b": "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-8b/20260328_210527/templeton_aut_create_short_eval_data_deepseek-r1-distill-llama-8b_20260328_210527.json",
    "deepseek-r1-distill-llama-70b": "../outputs/templeton_aut/aut/deepseek-r1-distill-llama-70b/20260328_211319/templeton_aut_create_short_eval_data_deepseek-r1-distill-llama-70b_20260328_211319.json",
    "falcon-40b-instruct": "../outputs/templeton_aut/aut/falcon-40b-instruct/20260328_214322/templeton_aut_create_short_eval_data_falcon-40b-instruct_20260328_214322.json",
    "qwen2.5-14b-instruct": "../outputs/templeton_aut/aut/qwen2.5-14b-instruct/20260328_214636/templeton_aut_create_short_eval_data_qwen2.5-14b-instruct_20260328_214636.json",
    "qwen2.5-7b-instruct": "../outputs/templeton_aut/aut/qwen2.5-7b-instruct/20260527_111143/templeton_aut_create_short_eval_data_qwen2.5-7b-instruct_20260527_111143.json"
}
# ---------------------------------------------------------------------------

ID_FIELD = "id"        # change if your items group by a different field name
OUTPUT_FIELD = "output"
GEMINI_AUT_SCORE_FIELD = "gemini_aut_score"

N_BOOT = 10_000        # bootstrap resamples for per-item mean CI
CI = 95.0              # confidence level for the interval
SEED = 0               # reproducibility
OUT_PATH = Path("experiments/templeton_aut/data/aut_score_stats.md")


def parse_score(s):
    """Extract a single float from the model output string. Gemini sometimes
    pads with whitespace, newlines, or stray text; we grab the first number."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None


def load_scores(path):
    """Return list of (item_id, score) pairs from one model's scoring file.
    Items whose score fails to parse are dropped and counted separately."""
    p = Path(path)
    if not p.exists():
        return None, f"file not found: {p}"
    data = json.loads(p.read_text())
    items = data.get("data", [])
    parsed, dropped = [], 0
    for it in items:
        if "aut_scoring" in path:
            score = parse_score(it.get(OUTPUT_FIELD))
        else:
            score = parse_score(it.get(GEMINI_AUT_SCORE_FIELD))
        iid = it.get(ID_FIELD)
        if score is None or iid is None:
            dropped += 1
            continue
        parsed.append((iid, score))
    return parsed, dropped


def global_stats(pairs):
    vals = [s for _, s in pairs]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": st.fmean(vals),
        "median": st.median(vals),
        "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def per_item_stats(pairs):
    """Median across generations within each item_id, then mean/median/std
    across items. Controls for within-item noise before aggregating. Also
    returns the per-item-medians vector for downstream bootstrapping."""
    by_item = {}
    for iid, s in pairs:
        by_item.setdefault(iid, []).append(s)
    item_medians = [st.median(v) for v in by_item.values()]
    if not item_medians:
        return None
    return {
        "n_items": len(item_medians),
        "gens_per_item_mean": st.fmean(len(v) for v in by_item.values()),
        "mean": st.fmean(item_medians),
        "median": st.median(item_medians),
        "std": st.pstdev(item_medians) if len(item_medians) > 1 else 0.0,
        "item_medians": item_medians,
    }


def bootstrap_mean_ci(values, n_boot=N_BOOT, ci=CI, seed=SEED):
    """Bootstrap percentile CI for the mean, resampling the input vector
    with replacement. For per-item-mean comparisons across models, the
    input should be the vector of per-item medians: this captures
    variability across stimuli, which is the relevant uncertainty when
    comparing two models' overall AUT scores."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo_q, hi_q = (100 - ci) / 200, 1 - (100 - ci) / 200
    return boots[int(lo_q * n_boot)], boots[int(hi_q * n_boot) - 1]


def fmt(x, w=5, prec=2):
    return f"{x:.{prec}f}" if isinstance(x, float) else f"{x:>{w}}"


def main():
    rows = []
    notes = []
    for model, path in GEMINI_AUT_SCORING_PATHS.items():
        pairs, dropped_or_err = load_scores(path)
        if pairs is None:
            notes.append((model, dropped_or_err))
            continue
        g = global_stats(pairs)
        pi = per_item_stats(pairs)
        ci_lo, ci_hi = bootstrap_mean_ci(pi["item_medians"])
        rows.append({
            "model": model,
            "n_total": g["n"], "g_mean": g["mean"], "g_median": g["median"], "g_std": g["std"],
            "n_items": pi["n_items"], "gens_per_item": pi["gens_per_item_mean"],
            "pi_mean": pi["mean"], "pi_median": pi["median"], "pi_std": pi["std"],
            "pi_ci_lo": ci_lo, "pi_ci_hi": ci_hi,
            "dropped": dropped_or_err if isinstance(dropped_or_err, int) else 0,
        })

    # sort by per-item mean, descending -- highest-scoring models at the top
    rows.sort(key=lambda r: r["pi_mean"], reverse=True)

    # --- assemble markdown ---
    lines = []
    lines.append("## AUT Score Statistics (Gemini-3-Flash scoring)\n")
    lines.append(
        "Per-item aggregation = median across generations within each item, "
        "then statistics across items. This controls for within-item noise and "
        "is the primary number for cross-model comparison. The 95% CI is a "
        f"percentile bootstrap (N={N_BOOT}) resampling items with replacement, "
        "which captures variability across the stimulus set. Global stats "
        "(flattened over all generations) are reported for reference.\n"
    )
    hdr = (
        "| model | n_items | gens/item | "
        "**per-item mean** | per-item 95% CI | per-item median | per-item std | "
        "global mean | global median | global std | n_total | dropped |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines.append(hdr)
    lines.append(sep)
    for r in rows:
        lines.append(
            f"| `{r['model']}` "
            f"| {r['n_items']} | {r['gens_per_item']:.1f} "
            f"| **{r['pi_mean']:.3f}** "
            f"| [{r['pi_ci_lo']:.3f}, {r['pi_ci_hi']:.3f}] "
            f"| {r['pi_median']:.3f} | {r['pi_std']:.3f} "
            f"| {r['g_mean']:.3f} | {r['g_median']:.3f} | {r['g_std']:.3f} "
            f"| {r['n_total']} | {r['dropped']} |"
        )
    if notes:
        lines.append("\n**Notes (load errors):**")
        for m, n in notes:
            lines.append(f"- `{m}`: {n}")

    md = "\n".join(lines) + "\n"
    print(md)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(md)
    print(f"\n[wrote markdown -> {OUT_PATH}]")


if __name__ == "__main__":
    main()