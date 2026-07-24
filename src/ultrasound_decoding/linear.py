from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


def _safe_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = np.dot(a, b)
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite value produced during matrix multiplication")
    return out


def preprocess_frames(X: np.ndarray) -> np.ndarray:
    """Variance-stabilize and flatten [n, h, w] frames for linear decoders."""
    return np.arcsinh(X.astype(np.float64, copy=False)).reshape(len(X), -1)


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standard scaler is not fitted")
        return (X - self.mean_) / self.scale_


@dataclass
class PCATransformer:
    variance: float = 0.95
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    n_components_: int | None = None

    def fit(self, X: np.ndarray) -> "PCATransformer":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        eig = (s**2) / max(len(X) - 1, 1)
        total = eig.sum()
        if total <= 0:
            n_components = 1
        else:
            n_components = int(np.searchsorted(np.cumsum(eig) / total, self.variance) + 1)
        n_components = max(1, min(n_components, vt.shape[0]))
        self.components_ = vt[:n_components]
        self.n_components_ = n_components
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA transformer is not fitted")
        return _safe_dot(X - self.mean_, self.components_.T)


@dataclass
class ClassContrastivePCATransformer:
    variance: float = 0.95
    alpha: float = 1.0
    max_pre_components: int = 64
    mean_: np.ndarray | None = None
    pre_components_: np.ndarray | None = None
    cpca_components_: np.ndarray | None = None
    n_components_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassContrastivePCATransformer":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        pre_k = max(1, min(self.max_pre_components, vt.shape[0], len(X) - 1))
        self.pre_components_ = vt[:pre_k]
        Z = _safe_dot(Xc, self.pre_components_.T)

        classes = np.unique(y)
        grand_mean = Z.mean(axis=0, keepdims=True)
        between = np.zeros((pre_k, pre_k), dtype=np.float64)
        within = np.zeros((pre_k, pre_k), dtype=np.float64)
        for cls in classes:
            Zc = Z[y == cls]
            diff = Zc.mean(axis=0, keepdims=True) - grand_mean
            between += len(Zc) * _safe_dot(diff.T, diff)
            residual = Zc - Zc.mean(axis=0, keepdims=True)
            within += _safe_dot(residual.T, residual)

        contrast = between / max(len(X) - 1, 1) - self.alpha * within / max(len(X) - len(classes), 1)
        eigvals, eigvecs = np.linalg.eigh((contrast + contrast.T) / 2.0)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        positive = np.maximum(eigvals, 0.0)
        if positive.sum() > 0:
            n_components = int(np.searchsorted(np.cumsum(positive) / positive.sum(), self.variance) + 1)
        else:
            n_components = min(max(len(classes) - 1, 1), pre_k)
        n_components = max(1, min(n_components, pre_k))
        self.cpca_components_ = eigvecs[:, :n_components]
        self.n_components_ = n_components
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.pre_components_ is None or self.cpca_components_ is None:
            raise RuntimeError("cPCA transformer is not fitted")
        Z = _safe_dot(X - self.mean_, self.pre_components_.T)
        return _safe_dot(Z, self.cpca_components_)


@dataclass
class LDAModel:
    reg: float = 1e-3
    classes_: np.ndarray | None = None
    means_: np.ndarray | None = None
    priors_: np.ndarray | None = None
    inv_cov_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDAModel":
        self.classes_ = np.unique(y)
        means = []
        priors = []
        pooled = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        for cls in self.classes_:
            Xc = X[y == cls]
            mean = Xc.mean(axis=0)
            means.append(mean)
            priors.append(len(Xc) / len(X))
            residual = Xc - mean
            pooled += _safe_dot(residual.T, residual)
        pooled /= max(len(X) - len(self.classes_), 1)
        scale = np.trace(pooled) / max(pooled.shape[0], 1)
        pooled += np.eye(pooled.shape[0]) * self.reg * max(scale, 1e-12)
        self.means_ = np.vstack(means)
        self.priors_ = np.asarray(priors)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.inv_cov_ = np.linalg.pinv(pooled)
        if not np.isfinite(self.inv_cov_).all():
            raise FloatingPointError("Non-finite value produced while inverting LDA covariance")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.means_ is None or self.priors_ is None or self.inv_cov_ is None:
            raise RuntimeError("LDA model is not fitted")
        linear = _safe_dot(_safe_dot(X, self.inv_cov_), self.means_.T)
        quadratic = 0.5 * np.sum(_safe_dot(self.means_, self.inv_cov_) * self.means_, axis=1)
        scores = linear - quadratic + np.log(self.priors_)
        return self.classes_[np.argmax(scores, axis=1)]


def fit_predict_linear(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    pca_variance: float = 0.95,
    standardize: bool = False,
) -> tuple[np.ndarray, int]:
    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if method == "pca_lda":
        transformer = PCATransformer(variance=pca_variance).fit(X_train)
    elif method == "cpca_lda":
        transformer = ClassContrastivePCATransformer(variance=pca_variance).fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown linear method: {method}")

    Z_train = transformer.transform(X_train)
    Z_test = transformer.transform(X_test)
    model = LDAModel().fit(Z_train, y_train)
    return model.predict(Z_test), int(transformer.n_components_ or Z_train.shape[1])
