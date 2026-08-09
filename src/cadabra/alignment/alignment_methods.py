from abc import abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
import concurrent.futures
from functools import partial

from cadabra.alignment.alignment_utils import ModelSample, BrainSample, compute_alignment_metrics, apply_noise_ceiling, AlignmentResult
from cadabra.utils import keep_most_frequent_size, compute_rsa

class AlignmentMethod:
    @abstractmethod
    def compute(self, model_data: List[ModelSample], brain_data: List[BrainSample]):
        raise NotImplementedError
    
    def compute_metrics(self, alignment_scores: np.ndarray) -> dict:
        return compute_alignment_metrics(alignment_scores)
    
    def apply_noise_ceiling(self, alignment_results, noise_ceiling_results):
        return apply_noise_ceiling(alignment_results, noise_ceiling_results)

def prepare_model_neural_sample(model_samples: List[ModelSample]):
    """
    Prepare model neural sample by taking mean across samples if multiple samples.

    Args:
        model_samples: List[ModelSample], model neural data
    Returns:
        X: np.ndarray, shape (T, L, model_dim) - model neural sample
    """
    # Take mean across samples if multiple samples per stimulus
    model_neural_sample = np.mean(np.stack([ms.data for ms in model_samples]), axis=0)
    return model_neural_sample

def prepare_brain_neural_sample(brain_samples: List[BrainSample]):
    """
    Prepare brain neural sample by flattening and taking mean across samples if multiple samples.

    Args:
        brain_samples: List[BrainSample], brain neural data
    Returns:
        Y: np.ndarray, shape (brain_dim,) - brain neural sample
    """
    brain_neural_samples = [bs.data.flatten() for bs in brain_samples]
    # brain dimension could be different for different subjects, keep only the most frequent one
    brain_neural_samples = keep_most_frequent_size(brain_neural_samples, dim=-1)
    brain_neural_sample = np.mean(np.stack(brain_neural_samples), axis=0)
    return brain_neural_sample

def prepare_alignment_data(model_data: List[ModelSample], brain_data: List[BrainSample], match_subjects: bool = False):
    """
    Prepare alignment data by matching stimuli and flattening brain data.

    Args:
        model_data: List[ModelSample], model neural data
        brain_data: List[BrainSample], brain neural data
        match_subjects: bool, whether to match subjects between model and brain data
    Returns:
        X: List[np.ndarray], shape List(num_samples, model_dim) - model neural data
        Y: List[np.ndarray], shape List(num_samples, brain_dim) - brain neural data
    """
    model_stimuli = set([ms.stimuli for ms in model_data])
    brain_stimuli = set([bs.stimuli for bs in brain_data])
    common_stimuli = model_stimuli.intersection(brain_stimuli)
    print(f"Found {len(common_stimuli)} common stimuli between model and brain data.")

    X_data = []
    Y_data = []

    for stim in tqdm(common_stimuli, desc="Preparing alignment data"):
        if match_subjects:
            model_samples = [ms for ms in model_data if ms.stimuli == stim and ms.subject is not None and ms.subject != "N/A"]
            brain_samples = [bs for bs in brain_data if bs.stimuli == stim and bs.subject is not None and bs.subject != "N/A"]
            common_subjects = list(set([ms.subject for ms in model_samples]).intersection(set([bs.subject for bs in brain_samples])))
            print(f"Found {len(common_subjects)} common subjects for stimuli {stim}.")

            for subject in common_subjects:
                subject_model_samples = [ms for ms in model_samples if ms.subject == subject]
                subject_brain_samples = [bs for bs in brain_samples if bs.subject == subject]

                if len(subject_model_samples) == 0 or len(subject_brain_samples) == 0:
                    continue

                model_neural_sample = prepare_model_neural_sample(subject_model_samples)
                brain_neural_sample = prepare_brain_neural_sample(subject_brain_samples)

                X_data.append(model_neural_sample)
                Y_data.append(brain_neural_sample)
        else:
            model_samples = [ms for ms in model_data if ms.stimuli == stim]
            brain_samples = [bs for bs in brain_data if bs.stimuli == stim]

            if len(model_samples) == 0 or len(brain_samples) == 0:
                continue

            model_neural_sample = prepare_model_neural_sample(model_samples)
            brain_neural_sample = prepare_brain_neural_sample(brain_samples)

            X_data.append(model_neural_sample)
            Y_data.append(brain_neural_sample)
    
    try:
        X = np.stack(X_data)
    except ValueError:
        print("Model data has inconsistent dimensions across samples likely due to the varying number of response tokens."
              "Keeping up to the minimum dimension size.")
        dim_sizes = [x.shape[0] for x in X_data]
        min_dim = min(dim_sizes)
        X_data = [x[:min_dim] for x in X_data]
        X = np.stack(X_data)

    try:
        Y = np.stack(Y_data)
    except ValueError:
        print("Brain data has inconsistent dimensions across samples. Keeping only the most frequent size.")
        Y_data, frequent_indices = keep_most_frequent_size(Y_data, dim=-1, return_indices=True)
        Y = np.stack(Y_data)
        # also remove corresponding samples from X
        X = X[frequent_indices]
    
    return X, Y

def run_linear_regression(X: np.ndarray, Y: np.ndarray, n_splits=5, random_seed=42, ridge=False, alphas=[0.1, 1.0, 10.0], **kwargs) -> np.ndarray:
    """
    Run cross-validated linear regression to compute predictivity.
    Args:
        X: np.ndarray, shape (num_samples, model_dim) - model neural data
        Y: np.ndarray, shape (num_samples, brain_dim) - brain neural data
        n_splits: int, number of cross-validation splits
        random_seed: int, random seed for reproducibility
        ridge: bool, whether to use ridge regression
        alphas: List[float], list of alphas for ridge regression
    Returns:
        predictivity: np.ndarray, shape (brain_dim,) - Pearson correlation per brain signal unit
    """
    if X.shape[0] < n_splits:
        print(f"Skipping due to insufficient samples ({X.shape[0]} samples).")
        return np.zeros(Y.shape[1])

    print(f"Model data shape: {X.shape}, Brain data shape: {Y.shape}")
    assert X.shape[0] == Y.shape[0], "Number of samples must match between X and Y."

    # check if there are any NaN values in X or Y
    if np.isnan(X).any() or np.isnan(Y).any():
        # remove samples with NaN values
        print("Found NaN values in data. Removing samples with NaN values.")
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
        X = X[valid_indices]
        Y = Y[valid_indices]
        print(f"New data shape after removing NaNs: Model data shape: {X.shape}, Brain data shape: {Y.shape}")
        if X.shape[0] < n_splits:
            print(f"Skipping due to insufficient samples ({X.shape[0]} samples) after removing NaNs.")
            return np.zeros(Y.shape[1])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    corrs_all_folds = []

    for train_idx, test_idx in tqdm(kf.split(X), total=n_splits, desc=f"Running cross-validation", position=1):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Fit linear/ridge regression with cross-validated alpha
        model = LinearRegression()
        if ridge:
            model = RidgeCV(alphas=alphas)
        
        model.fit(X_train, Y_train)

        # Predict on held-out data
        Y_pred = model.predict(X_test)

        # if one-dimensional output, reshape to 2D
        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)

        # Compute Pearson correlation per brain signal unit
        corrs = []
        for v in range(Y.shape[-1]):
            y_true = Y_test[:, v]
            y_pred = Y_pred[:, v]
            if np.std(y_true) < 1e-6:
                corrs.append(0.0)
                continue
            r, _ = pearsonr(y_true, y_pred)
            corrs.append(r)
        corrs_all_folds.append(corrs)
    predictivity = np.mean(corrs_all_folds, axis=0)
    return predictivity

def run_per_voxel_linear_regression(X: np.ndarray, Y: np.ndarray, **kwargs):
    # run per voxel regression
    per_voxel_predictivities = np.zeros(Y.shape[-1])
    for voxel in range(Y.shape[-1]):
        voxel_Y = Y[:, voxel].reshape(-1, 1)
        voxel_predictivity = run_linear_regression(X, voxel_Y, **kwargs)
        per_voxel_predictivities[voxel] = np.mean(voxel_predictivity)
    return per_voxel_predictivities

def compute_lr_predictivity(X: np.ndarray, Y: np.ndarray, regression_target: Optional[str] = None, **kwargs) -> np.ndarray:
    """
    Compute linear regression predictivity with different regression targets.
    Args:
        X: np.ndarray, shape (num_samples, model_dim) - model neural data
        Y: np.ndarray, shape (num_samples, brain_dim) - brain neural data
        regression_target: str, regression target to compute predictivity for. 
                                Options: "per_voxel", "mean_voxel", "median_voxel". 
                                If None, compute predictivity using all voxels.
    Returns:
        linear_predictivity: np.ndarray, shape (brain_dim,) - Pearson correlation per brain signal unit
    """
    if regression_target == "per_voxel":
        print("Computing per-voxel linear regression predictivity.")
        linear_predictivity = run_per_voxel_linear_regression(X, Y, **kwargs)
    elif regression_target == "mean_voxel":
        print("Computing mean-voxel linear regression predictivity.")
        mean_Y = np.mean(Y, axis=-1).reshape(-1, 1)
        linear_predictivity = run_linear_regression(X, mean_Y, **kwargs)
    elif regression_target == "median_voxel":
        print("Computing median-voxel linear regression predictivity.")
        median_Y = np.median(Y, axis=-1).reshape(-1, 1)
        linear_predictivity = run_linear_regression(X, median_Y, **kwargs)
    else:
        linear_predictivity = run_linear_regression(X, Y, **kwargs)
    return linear_predictivity

def compute_rsa_predictivity(X: np.ndarray, Y: np.ndarray, **kwargs):
    """
    Compute RSA predictivity.
    Args:
        X: np.ndarray, shape (num_samples, model_dim) - model neural data
        Y: np.ndarray, shape (num_samples, brain_dim) - brain neural data
        kwargs: additional arguments for RSA computation
    Returns:
        rsa_predictivity: np.ndarray, shape (1,) - RSA correlation
    """
    rsa, _ = compute_rsa(X, Y, **kwargs)
    return np.array([rsa])

def compute_predictivity_over_layers(X: np.ndarray, Y: np.ndarray, method=compute_lr_predictivity, **kwargs) -> np.ndarray:
    """
    Compute predictivity per layer.
    Args:
        X: np.ndarray, shape (num_samples, num_layers, model_dim) - model neural data
        Y: np.ndarray, shape (num_samples, brain_dim) - brain neural data
        kwargs: additional arguments for the alignment method
    Returns:
        linear_predictivity: np.ndarray, shape (num_layers, brain_dim) - Pearson correlation per brain signal unit
    """
    print(f"Model data has layer dimension with {X.shape[1]} layers. Computing predictivity per layer.")
    all_layer_predictivities = []
    for layer in range(X.shape[1]):
        print(f"Processing layer {layer}/{X.shape[1]}")
        layer_X = X[:, layer, :]  # (num_samples, model_dim)
        layer_predictivity = method(layer_X, Y, **kwargs)
        all_layer_predictivities.append(layer_predictivity)
    all_layer_predictivities = np.stack(all_layer_predictivities, axis=0)  # (num_layers, brain_dim)
    return all_layer_predictivities

def compute_predictivity_over_tokens(X: np.ndarray, Y: np.ndarray, method=compute_lr_predictivity, **kwargs) -> np.ndarray:
    """
    Compute predictivity per token.
    Args:
        X: np.ndarray, shape (num_samples, num_tokens, num_layers, model_dim) - model neural data
        Y: np.ndarray, shape (num_samples, brain_dim) - brain neural data
        kwargs: additional arguments for the alignment method
    Returns:
        linear_predictivity: np.ndarray, shape (num_tokens, num_layers, brain_dim) - Pearson correlation per brain signal unit
    """
    print(f"Model data has token dimension with {X.shape[1]} tokens. Computing predictivity per token.")
    all_token_predictivities = []
    for token in range(X.shape[1]):
        print(f"Processing token {token}/{X.shape[1]}")
        token_X = X[:, token, :, :]  # (num_samples, num_layers, model_dim)
        token_predictivity = compute_predictivity_over_layers(token_X, Y, method=method, **kwargs)
        all_token_predictivities.append(token_predictivity)
    all_token_predictivities = np.stack(all_token_predictivities, axis=0)  # (num_tokens, num_layers, brain_dim)
    return all_token_predictivities

class BaseAlignment(AlignmentMethod):
    def __init__(self, alignment_method, alignment_args, match_subjects=False):
        self.alignment_method = alignment_method
        self.alignment_args = alignment_args
        self.match_subjects = match_subjects

    def compute(self, model_data: List[ModelSample], brain_data: List[BrainSample]):
        """
        Compute global predictivity. 

        Args:
            model_data: List[ModelSample], model alignment data
            brain_data: List[BrainSample], brain alignment data

        Returns:
            predictivity: np.ndarray, shape (num_tokens, num_layers, brain_dim) - Correlation per token per layer per brain signal unit
        """
        X, Y = prepare_alignment_data(model_data, brain_data, match_subjects=self.match_subjects)
        predictivity = compute_predictivity_over_tokens(X, Y, method=self.alignment_method, **self.alignment_args)
        return AlignmentResult(alignment_scores=predictivity)

class BasePerSubjectAlignment(AlignmentMethod):
    def __init__(self, alignment_method, alignment_args, subject_sampling=False, 
                 num_subjects=10, in_parallel=False, random_seed=42):
        self.alignment_method = alignment_method
        self.alignment_args = alignment_args
        self.subject_sampling = subject_sampling
        self.num_subjects = num_subjects
        self.in_parallel = in_parallel
        self.random_seed = random_seed

    def compute(self, model_data: List[ModelSample], brain_data: List[BrainSample]):
        """
        Compute predictivity per subject.

        Args:
            model_data: List[ModelSample], model alignment data
            brain_data: List[BrainSample], brain alignment data

        Returns:
            predictivity: np.ndarray, shape (num_tokens, num_layers, num_subjects) - Predictivity per token per layer per subject.
        """
        subjects = set([bs.subject for bs in brain_data])
        print(f"Found {len(subjects)} subjects in brain data.")

        if self.subject_sampling and len(subjects) > self.num_subjects:
            print(f"Sampling {self.num_subjects} subjects from {len(subjects)} available subjects.")
            np.random.seed(self.random_seed)
            subjects = np.random.choice(list(subjects), size=self.num_subjects, replace=False)
            print(f"Selected subjects: {subjects}")

        subject_predictivities = []
        subjects = list(subjects)

        if self.in_parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                subject_predictivities = list(tqdm(executor.map(partial(self._process_subject, model_data=model_data, brain_data=brain_data), subjects),
                                              total=len(subjects), desc="Processing subjects (parallel)", position=0))
        else:
            for subject in tqdm(subjects, desc="Processing subjects", leave=False, position=0):
                subject_predictivity = self._process_subject(subject, model_data, brain_data)
                subject_predictivities.append(subject_predictivity)
        
        try:
            subject_predictivities = np.stack(subject_predictivities, axis=0)  # (num_subjects, num_tokens, num_layers, brain_dim)
        except ValueError as e:
            print(f"Number of tokens is inconsistent across subjects. Keeping up to the minimum number of tokens.")
            min_tokens = min([sp.shape[0] for sp in subject_predictivities])
            subject_predictivities = [sp[:min_tokens] for sp in subject_predictivities]
            subject_predictivities = np.stack(subject_predictivities, axis=0) # (num_subjects, min_num_tokens, num_layers, brain_dim)
        
        print(f"Computed subject predictivities with shape {subject_predictivities.shape}.")
        # take median across voxels to compute subject score per layer
        alignment_scores = np.nanmedian(subject_predictivities, axis=-1)  # shape (num_subjects, num_tokens, num_layers)
        print(f"Computed alignment scores with shape {alignment_scores.shape} by taking median across brain signal units.")
        # reshape to (num_tokens, num_layers, num_subjects)
        alignment_scores = alignment_scores.transpose(1, 2, 0)
        print(f"Reshaped alignment scores to shape {alignment_scores.shape}")
        return AlignmentResult(alignment_scores=alignment_scores, subject_alignment_scores=subject_predictivities, subjects=subjects)
       
    def _process_subject(self, subject, model_data, brain_data):
        print(f"Processing subject: {subject}")
        subject_brain_data = [bs for bs in brain_data if bs.subject == subject]
        subject_model_data = [ms for ms in model_data if (ms.subject == subject) or (ms.subject is None) or (ms.subject == "N/A")]
        
        if len(subject_brain_data) == 0:
            return np.zeros((1,))  # No data for this subject

        X, Y = prepare_alignment_data(subject_model_data, subject_brain_data)
        subject_predictivity = compute_predictivity_over_tokens(X, Y, method=self.alignment_method, **self.alignment_args)
        return subject_predictivity
    
class LinearRegressionAlignment(BaseAlignment):
    def __init__(self, n_splits=5, random_seed=42, ridge=True, alphas=[0.1, 1.0, 10.0], match_subjects=False, regression_target=None):
        alignment_args = {
            "n_splits": n_splits,
            "random_seed": random_seed,
            "ridge": ridge,
            "alphas": alphas,
            "regression_target": regression_target,
            "match_subjects": match_subjects
        }
        super().__init__(compute_lr_predictivity, alignment_args, match_subjects=match_subjects)

    def compute(self, model_data: List[ModelSample], brain_data: List[BrainSample]):
        """
        Linear predictivity using cross-validated ordinary least-squares linear regression.

        Args:
            model_data: List[ModelSample], model alignment data
            brain_data: List[BrainSample], brain alignment data

        Returns:
            linear_predictivity: np.ndarray, shape (num_tokens, num_layers, brain_dim) - Pearson correlation per brain signal unit
        """
        print(f"Computing linear predictivity with {self.alignment_args['n_splits']}-fold cross-validation.")
        return super().compute(model_data, brain_data)

class LinearRegressionPerSubjectAlignment(BasePerSubjectAlignment):
    def __init__(self, n_splits=5, random_seed=42, ridge=True, alphas=[0.1, 1.0, 10.0], 
                 subject_sampling=False, num_subjects=10, in_parallel=False, regression_target=None):
        alignment_args = {
            "n_splits": n_splits,
            "random_seed": random_seed,
            "ridge": ridge,
            "alphas": alphas,
            "regression_target": regression_target
        }
        super().__init__(compute_lr_predictivity, alignment_args, 
                         subject_sampling=subject_sampling, num_subjects=num_subjects, 
                         in_parallel=in_parallel, random_seed=random_seed)

    def compute(self, model_data: List[ModelSample], brain_data: List[BrainSample]):
        """
        Linear regression predictivity using cross-validated linear regression per subject.

        Args:
            model_data: List[ModelSample], model alignment data
            brain_data: List[BrainSample], brain alignment data

        Returns:
            linear_predictivity: np.ndarray, shape (num_tokens, num_layers, num_subjects) - Pearson correlation per token per layer per subject
        """
        print(f"Computing linear regression predictivity per subject with {self.alignment_args['n_splits']}-fold cross-validation over alphas {self.alignment_args['alphas']}.")
        return super().compute(model_data, brain_data)

class RSAAlignment(BaseAlignment):
    def __init__(self, random_seed=42, match_subjects=False):
        alignment_args = {
            "random_seed": random_seed,
        }
        super().__init__(compute_rsa_predictivity, alignment_args, match_subjects=match_subjects)

    def compute(self, model_data: List[ModelSample], brain_data: List[BrainSample]):
        """
        RSA predictivity.

        Args:
            model_data: List[ModelSample], model alignment data
            brain_data: List[BrainSample], brain alignment data

        Returns:
            rsa_predictivity: np.ndarray, shape (num_tokens, num_layers) - RSA correlation per token per layer.
        """
        print(f"Computing RSA predictivity")
        return super().compute(model_data, brain_data)

class RSAPerSubjectAlignment(BasePerSubjectAlignment):
    def __init__(self, random_seed=42, subject_sampling=False, num_subjects=10, in_parallel=False):
        alignment_args = {
            "random_seed": random_seed,
        }
        super().__init__(compute_rsa_predictivity, alignment_args, subject_sampling=subject_sampling, 
                         num_subjects=num_subjects, in_parallel=in_parallel, random_seed=random_seed)

    def compute(self, model_data: List[ModelSample], brain_data: List[BrainSample]):
        """
        RSA predictivity per subject.

        Args:
            model_data: List[ModelSample], model alignment data
            brain_data: List[BrainSample], brain alignment data

        Returns:
            rsa_predictivity: np.ndarray, shape (num_tokens, num_layers, num_subjects) - RSA correlation per token per layer per subject.
        """
        print(f"Computing RSA predictivity per subject.")
        return super().compute(model_data, brain_data)