"""
Cross-validated layer selection: how much of the best-layer alignment score is
optimism bias, and does the size correlation survive removing it?

MOTIVATION
----------
Reported alignment takes the MAXIMUM across layers of each model. The expected
maximum over more candidates is higher, so this estimator is biased upward and
the bias grows with layer count. Layer count is strongly collinear with model
size, which creates a mechanical path from size to alignment that is independent
of any representational claim. Since the DMN-size correlation is the primary
result that survives correction, this needs quantifying.

METHOD
------
Naive (as reported):
    score = max_L  median_over_all_subjects( A[L, s] )

Cross-validated (unbiased w.r.t. layer selection):
    repeated K-fold over SUBJECTS. For each fold f:
        L_f = argmax_L median_over_subjects_NOT_in_f( A[L, s] )
        collect A[L_f, s] for s in f          <- held-out evaluation
    score = median over all subjects of their held-out score,
            averaged over repeats.
Selection and evaluation therefore never share subjects, so the maximum
operation cannot inflate the evaluated value.

We then compare:
  (a) naive vs CV alignment per model  -> the optimism gap, and whether it
      scales with layer count as the confound predicts;
  (b) corr(size, alignment) under both estimators -> does the headline
      correlation survive;
  (c) corr(size, n_layers) and the partial correlation of size with alignment
      controlling for n_layers. NOTE: if size and n_layers are highly
      collinear, the partial correlation removes most of size's variance and
      is close to uninterpretable -- we report the collinearity so the reader
      can judge.

INPUT
-----
main_results.json entries -> get_alignment_runs(...) -> run.config['output_path']
-> alignment metadata json -> data.noise_ceiling_adjusted_scores_path
-> numpy array of shape (num_tokens=1, num_layers, num_subjects).
"""

import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from cadabra.utils import read_json
from cadabra.alignment.alignment_utils import get_alignment_runs   # adjust import if needed

# ---------------------------------------------------------------------------
RESULTS_JSON = Path("experiments/templeton_aut/data/main_results.json")
OUTDIR = Path("experiments/templeton_aut/data")

# The cell to analyse: the primary DMN-size correlation.
TARGET = {
    "brain_network_contains": "yeo_dmn.*dt_create_with_ratings.json",
    "brain_task": "AUT",          # brain_network contains "create"
    "model_task": "AUT",          # dataset contains "create"
    "stage": "prompt",            # activation_mode
    "pooling": "last_token",      # model_data_sampling contains "last"
    "prompt_variant": "main",     # dataset does NOT contain "short_eval"
}

K_FOLDS = 5
N_REPEATS = 20
RNG = np.random.default_rng(0)
# ---------------------------------------------------------------------------


def matches_target(entry) -> bool:
    bn, ds = entry["brain_network"], entry["dataset"]
    if TARGET["brain_network_contains"] not in bn:
        return False
    if ("AUT" if "create" in bn else "OCT") != TARGET["brain_task"]:
        return False
    if ("AUT" if "create" in ds else "OCT") != TARGET["model_task"]:
        return False
    if entry["activation_mode"] != TARGET["stage"]:
        return False
    pooling = "mean_token" if "mean" in entry["model_data_sampling"] else "last_token"
    if pooling != TARGET["pooling"]:
        return False
    variant = "scoring" if "short_eval" in ds else "main"
    if variant != TARGET["prompt_variant"]:
        return False
    return entry["nc_threshold"] == 0


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
        activation_mode=entry["activation_mode"],
        force_refresh=True
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


def naive_best_layer(A):
    """max over layers of the median across ALL subjects (as reported)."""
    per_layer = np.nanmedian(A, axis=1)
    L = int(np.nanargmax(per_layer))
    return float(per_layer[L]), L


def cv_best_layer(A, k=K_FOLDS, repeats=N_REPEATS, rng=RNG):
    """Repeated K-fold over subjects. Select the layer on the training subjects,
    evaluate on the held-out subjects, aggregate. Returns the mean over repeats
    of the median held-out score, plus the spread across repeats."""
    n_layers, n_subj = A.shape
    if n_subj < k:
        return float("nan"), float("nan"), []

    per_repeat = []
    chosen_layers = []
    for _ in range(repeats):
        order = rng.permutation(n_subj)
        folds = np.array_split(order, k)
        held = np.full(n_subj, np.nan)
        for f in folds:
            mask = np.ones(n_subj, bool)
            mask[f] = False
            train_med = np.nanmedian(A[:, mask], axis=1)
            L = int(np.nanargmax(train_med))
            chosen_layers.append(L)
            held[f] = A[L, f]                  # held-out subjects only
        per_repeat.append(np.nanmedian(held))
    return float(np.mean(per_repeat)), float(np.std(per_repeat)), chosen_layers


def partial_corr(x, y, z):
    """Pearson correlation of x and y controlling for z, via residuals."""
    x, y, z = map(lambda v: np.asarray(v, float), (x, y, z))
    def resid(a):
        Z = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(Z, a, rcond=None)
        return a - Z @ beta
    rx, ry = resid(x), resid(y)
    r, p = pearsonr(rx, ry)
    return float(r), float(p)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    entries = [e for e in read_json(str(RESULTS_JSON)) if matches_target(e)]
    print(f"matched {len(entries)} entries for the target cell")

    rows = []
    for e in entries:
        model = e["model_name"]
        print(f"  fetching {model} ...")
        try:
            A = fetch_scores(e)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue
        naive, L_naive = naive_best_layer(A)
        cv, cv_sd, chosen = cv_best_layer(A)
        rows.append({
            "model": model,
            "n_layers": A.shape[0],
            "n_subjects": A.shape[1],
            "size_B": e["model_size_b"],
            "aut_score": e.get("aut_score", np.nan),
            "alignment_naive": naive,
            "alignment_cv": cv,
            "cv_sd_across_repeats": cv_sd,
            "optimism_gap": naive - cv,
            "best_layer_naive": L_naive,
            "best_layer_rel_depth": L_naive / max(A.shape[0] - 1, 1),
            "cv_layer_stability": float(np.std(chosen)) if chosen else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("size_B").reset_index(drop=True)
    df.to_csv(OUTDIR / "cv_layer_selection.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 96)
    print("PER-MODEL: naive vs cross-validated best-layer alignment")
    print("=" * 96)
    print(df[["model", "size_B", "n_layers", "alignment_naive", "alignment_cv",
              "optimism_gap", "best_layer_rel_depth"]].to_string(index=False))

    x_size = df["size_B"].to_numpy()
    x_lay = df["n_layers"].to_numpy(float)
    y_naive = df["alignment_naive"].to_numpy()
    y_cv = df["alignment_cv"].to_numpy()
    gap = df["optimism_gap"].to_numpy()

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"  median optimism gap (naive - CV): {np.median(gap):+.4f}"
          f"   range [{gap.min():+.4f}, {gap.max():+.4f}]")

    r, p = pearsonr(x_lay, gap)
    rs, ps = spearmanr(x_lay, gap)
    print(f"  gap vs n_layers:        r={r:+.3f} (p={p:.4f})  rho={rs:+.3f} (p={ps:.4f})")
    print("    (the confound predicts a POSITIVE relationship here)")

    r, p = pearsonr(x_size, x_lay)
    rs, ps = spearmanr(x_size, x_lay)
    print(f"\n  size vs n_layers:       r={r:+.3f} (p={p:.4f})  rho={rs:+.3f} (p={ps:.4f})")
    if abs(r) > 0.85:
        print("    NOTE: size and n_layers are highly collinear. The partial")
        print("    correlation below removes most of size's variance and should")
        print("    NOT be read as the decisive test; prefer the CV comparison.")

    rn, pn = pearsonr(x_size, y_naive)
    rc, pc = pearsonr(x_size, y_cv)
    print(f"\n  size vs alignment (naive): r={rn:+.3f} (p={pn:.4f})")
    print(f"  size vs alignment (CV)   : r={rc:+.3f} (p={pc:.4f})   <- key comparison")

    rp, pp = partial_corr(x_size, y_naive, x_lay)
    rpc, ppc = partial_corr(x_size, y_cv, x_lay)
    print(f"\n  partial corr size~alignment | n_layers (naive): r={rp:+.3f} (p={pp:.4f})")
    print(f"  partial corr size~alignment | n_layers (CV)   : r={rpc:+.3f} (p={ppc:.4f})")

    summary = pd.DataFrame([{
        "median_optimism_gap": float(np.median(gap)),
        "corr_gap_nlayers_r": float(pearsonr(x_lay, gap)[0]),
        "corr_gap_nlayers_p": float(pearsonr(x_lay, gap)[1]),
        "corr_size_nlayers_r": float(pearsonr(x_size, x_lay)[0]),
        "corr_size_alignment_naive_r": rn, "corr_size_alignment_naive_p": pn,
        "corr_size_alignment_cv_r": rc, "corr_size_alignment_cv_p": pc,
        "partial_size_alignment_cv_r": rpc, "partial_size_alignment_cv_p": ppc,
        "n_models": len(df),
    }])
    summary.to_csv(OUTDIR / "cv_layer_selection_summary.csv", index=False)
    print(f"\n  wrote {OUTDIR / 'cv_layer_selection.csv'}")
    print(f"  wrote {OUTDIR / 'cv_layer_selection_summary.csv'}")


if __name__ == "__main__":
    main()