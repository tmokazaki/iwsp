"""
Cross-Validated Ridge Regression Readout for Transparent Brain Encoding (arXiv:2611.18114)
==========================================================================================

In neuroscience brain encoding, an encoding model maps sensory or cognitive stimuli $S$
to recorded neural activity $Y \in \mathbb{R}^{N_{\text{samples}} \times N_{\text{voxels}}}$
via:
1. An agent-programmed, human-interpretable feature space $\Phi(S) \in \mathbb{R}^{N_{\text{samples}} \times D}$
   (written directly in Python code via iWSP without end-to-end black-box neural network training),
2. A standardized linear Ridge Regression readout $W = (\Phi^T \Phi + \alpha I)^{-1} \Phi^T Y$
   evaluated on held-out stimuli via voxel-wise Pearson correlation $r$.

This module also provides feature-group variance partitioning / importance attribution,
allowing researchers to inspect exactly which programmed features drive neural responses
in different brain regions (e.g., V1 vs V4/IT, or lexical semantics vs scan-edge timing
confounds in language fMRI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from iwsp.math_backend import np, pearson_corr


@dataclass
class RidgeEncodingReadout:
    """
    Standardized Ridge Regression readout with L2 regularization $\alpha$ selection
    and feature-group attribution.
    """

    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0)
    best_alpha: float = 1.0
    W: Any = None  # (n_features, n_targets)
    b: Any = None  # (n_targets,)
    x_mean: Any = None
    x_std: Any = None
    feature_names: List[str] = field(default_factory=list)
    region_names: List[str] = field(default_factory=list)

    def _standardize_fit(self, X: Any) -> Any:
        arr = np.asarray(X, dtype=np.float64)
        self.x_mean = arr.mean(axis=0, keepdims=True)
        self.x_std = arr.std(axis=0, keepdims=True) + 1e-8
        return (arr - self.x_mean) / self.x_std

    def _standardize_transform(self, X: Any) -> Any:
        arr = np.asarray(X, dtype=np.float64)
        return (arr - self.x_mean) / self.x_std

    @staticmethod
    def _solve_ridge(X_std: Any, Y_centered: Any, alpha: float) -> Any:
        n_feat = X_std.shape[1]
        gram = (X_std.T @ X_std) + (alpha * np.eye(n_feat, dtype=np.float64))
        rhs = X_std.T @ Y_centered
        return np.linalg.solve(gram, rhs)

    def fit(
        self,
        X_train: Any,
        Y_train: Any,
        feature_names: Optional[List[str]] = None,
        region_names: Optional[List[str]] = None,
    ) -> "RidgeEncodingReadout":
        """
        Fit Ridge Regression weights $W$ on `(X_train, Y_train)` using internal 2-fold
        split to pick the best regularization $\alpha \in \text{self.alphas}$.
        """
        X_arr = np.asarray(X_train, dtype=np.float64)
        Y_arr = np.asarray(Y_train, dtype=np.float64)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(Y_arr.shape[0], 1)

        n_samples, n_feat = X_arr.shape
        n_targets = Y_arr.shape[1]
        self.feature_names = feature_names or [f"feat_{i}" for i in range(n_feat)]
        self.region_names = region_names or [f"voxel_{j}" for j in range(n_targets)]

        # Select alpha via internal validation split
        mid = max(n_samples // 2, 2)
        X_sub_tr, X_sub_va = X_arr[:mid], X_arr[mid:]
        Y_sub_tr, Y_sub_va = Y_arr[:mid], Y_arr[mid:]

        sub_mean = X_sub_tr.mean(axis=0, keepdims=True)
        sub_std = X_sub_tr.std(axis=0, keepdims=True) + 1e-8
        X_tr_s = (X_sub_tr - sub_mean) / sub_std
        X_va_s = (X_sub_va - sub_mean) / sub_std
        y_tr_mean = Y_sub_tr.mean(axis=0, keepdims=True)
        Y_tr_c = Y_sub_tr - y_tr_mean

        best_r = float("-inf")
        best_a = float(self.alphas[0])
        for alpha in self.alphas:
            W_cand = self._solve_ridge(X_tr_s, Y_tr_c, float(alpha))
            Y_va_pred = (X_va_s @ W_cand) + y_tr_mean
            r_vec = pearson_corr(Y_sub_va, Y_va_pred, axis=0)
            mean_r = float(r_vec.mean())
            if mean_r > best_r:
                best_r = mean_r
                best_a = float(alpha)

        self.best_alpha = best_a
        X_full_s = self._standardize_fit(X_arr)
        self.b = Y_arr.mean(axis=0, keepdims=True)
        self.W = self._solve_ridge(X_full_s, Y_arr - self.b, self.best_alpha)
        return self

    def predict(self, X_test: Any) -> Any:
        """Predict neural responses for `X_test`."""
        X_s = self._standardize_transform(X_test)
        return (X_s @ self.W) + self.b

    def score(self, X_test: Any, Y_test: Any) -> Tuple[float, Dict[str, float]]:
        """
        Return `(mean_pearson_r, per_region_r_dict)` on held-out test data.
        """
        Y_arr = np.asarray(Y_test, dtype=np.float64)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(Y_arr.shape[0], 1)
        Y_pred = self.predict(X_test)
        r_vec = pearson_corr(Y_arr, Y_pred, axis=0)
        r_list = r_vec.tolist() if hasattr(r_vec, "tolist") else [float(r_vec)]
        if isinstance(r_list, float):
            r_list = [r_list]
        per_region = {
            name: round(float(r_val), 4)
            for name, r_val in zip(self.region_names, r_list)
        }
        mean_r = float(sum(r_list) / max(len(r_list), 1))
        return mean_r, per_region

    def top_features_per_region(self, top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        Return the top-k most influential interpretable features (by standardized Ridge weight
        magnitude) for each brain region / voxel group.
        """
        result: Dict[str, List[Tuple[str, float]]] = {}
        W_list = self.W.tolist() if hasattr(self.W, "tolist") else self.W
        for r_idx, r_name in enumerate(self.region_names):
            col_weights = [
                (self.feature_names[f_idx], round(float(W_list[f_idx][r_idx]), 4))
                for f_idx in range(len(self.feature_names))
            ]
            col_weights.sort(key=lambda item: abs(item[1]), reverse=True)
            result[r_name] = col_weights[:top_k]
        return result


class BrainEncodingEvaluator:
    """
    Evaluates an iWSP feature-extraction function `extract_features(stimuli) -> (X, feature_names)`
    against train/test brain recording splits using `RidgeEncodingReadout`.
    """

    def __init__(
        self,
        stimuli_train: Any,
        Y_train: Any,
        stimuli_test: Any,
        Y_test: Any,
        region_names: List[str],
    ) -> None:
        self.stimuli_train = stimuli_train
        self.Y_train = Y_train
        self.stimuli_test = stimuli_test
        self.Y_test = Y_test
        self.region_names = region_names

    def evaluate(
        self, feature_fn: Callable[[Any], Tuple[Any, List[str]]]
    ) -> Tuple[float, Dict[str, Any], str, RidgeEncodingReadout]:
        X_train, feat_names = feature_fn(self.stimuli_train)
        X_test, _ = feature_fn(self.stimuli_test)

        readout = RidgeEncodingReadout()
        readout.fit(
            X_train,
            self.Y_train,
            feature_names=feat_names,
            region_names=self.region_names,
        )
        mean_r, per_region = readout.score(X_test, self.Y_test)
        top_feats = readout.top_features_per_region(top_k=3)

        summary_parts: List[str] = []
        for reg, feats in top_feats.items():
            f_str = ", ".join(f"{fn}({w:+.2f})" for fn, w in feats)
            summary_parts.append(f"{reg} [r={per_region[reg]:.3f}]: {f_str}")
        interp_summary = " | ".join(summary_parts)

        metrics = {
            "mean_pearson_r": round(mean_r, 4),
            "num_features": len(feat_names),
            "best_alpha": readout.best_alpha,
            "region_r": per_region,
            "top_features": top_feats,
        }
        return mean_r, metrics, interp_summary, readout
