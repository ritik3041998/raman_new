"""Shared preprocessing for Raman spectra: resample -> smooth -> baseline -> normalise.

The same functions are used for training and for any spectrum predicted later,
so a spectrum from a new instrument goes through the identical steps.
"""
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

# Fingerprint region covered by every training source, 1 cm-1 step.
GRID = np.arange(450.0, 1801.0, 1.0)


def resample(x, y, grid=GRID):
    """Interpolate one spectrum onto the common grid. Returns None if it does not cover it."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 50 or x[0] > grid[0] + 10 or x[-1] < grid[-1] - 10:
        return None
    return np.interp(grid, x, y)


def als_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric least squares baseline (Eilers & Boelens)."""
    n = y.size
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    h = lam * (d.T @ d)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        z = spsolve((sparse.diags(w) + h).tocsc(), w * y)
        w = np.where(y > z, p, 1 - p)
    return z


def remove_baseline(X, smooth=9, n_jobs=-1):
    """Smooth, subtract the ALS baseline and clip negatives. X: (n, len(GRID))."""
    from joblib import Parallel, delayed

    def one(y):
        if smooth:
            y = savgol_filter(y, smooth, 3)
        return np.clip(y - als_baseline(y), 0, None)

    X = np.atleast_2d(np.asarray(X, float))
    if len(X) < 64:
        return np.array([one(y) for y in X])
    return np.array(Parallel(n_jobs=n_jobs, batch_size=64)(delayed(one)(y) for y in X))


def finish(B, sqrt=False, derivative=False, mask=None):
    """Optional square-root scaling / first derivative, keep selected features, L2-normalise."""
    B = np.atleast_2d(np.asarray(B, float))
    if sqrt:
        B = np.sqrt(np.clip(B, 0, None))
    if derivative:
        B = savgol_filter(B, 11, 3, deriv=1, axis=1)
    if mask is not None:
        B = B[:, mask]
    n = np.linalg.norm(B, axis=1, keepdims=True)
    return B / np.where(n > 0, n, 1.0)


def preprocess(X, baseline=True, sqrt=False, derivative=False, mask=None):
    """X: (n, len(GRID)) resampled intensities -> model-ready features."""
    X = np.atleast_2d(np.asarray(X, float))
    B = remove_baseline(X) if baseline else X - X.min(axis=1, keepdims=True)
    return finish(B, sqrt=sqrt, derivative=derivative, mask=mask)


class CosineMatcher:
    """Library search: score of a class = highest cosine similarity to any of its reference spectra."""

    def fit(self, F, y):
        self.classes_, self.idx_ = np.unique(y, return_inverse=True)
        self.lib_ = (F / np.linalg.norm(F, axis=1, keepdims=True)).astype(np.float32)  # halves the saved model
        return self

    def scores(self, F):
        F = F / np.linalg.norm(F, axis=1, keepdims=True)
        out = np.full((len(F), len(self.classes_)), -1.0)
        for s in range(0, len(F), 256):
            sim = F[s:s + 256] @ self.lib_.T
            np.maximum.at(out[s:s + 256].T, self.idx_, sim.T)
        return out

    def predict(self, F):
        return self.classes_[self.scores(F).argmax(1)]


class Ensemble:
    """Library match and a trained classifier, combined by averaging their standardised class scores."""

    def __init__(self, clf):
        self.clf = clf
        self.matcher = CosineMatcher()

    def fit(self, F, y):
        self.matcher.fit(F, y)
        self.classes_ = self.matcher.classes_
        self.clf.fit(F, self.matcher.idx_)
        return self

    def scores(self, F):
        def z(a):
            return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)
        return z(self.matcher.scores(F)) + z(self.clf.predict_log_proba(F))

    def predict(self, F):
        return self.classes_[self.scores(F).argmax(1)]


def augment(y, rng, grid=GRID):
    """One synthetic variant of a resampled raw spectrum: what another instrument might record."""
    span = y.max() - y.min()
    if span <= 0:
        return y.copy()
    y = (y - y.min()) / span
    # wavenumber calibration error: shift plus slight stretch
    shift = rng.uniform(-6, 6)
    stretch = 1 + rng.uniform(-0.004, 0.004)
    centre = grid.mean()
    y = np.interp(grid, (grid - centre) * stretch + centre + shift, y)
    # lower spectral resolution
    sigma = rng.uniform(0, 4)
    if sigma > 0.3:
        y = gaussian_filter1d(y, sigma)
    # fluorescence-like background
    t = np.linspace(-1, 1, grid.size)
    y = y + rng.uniform(0, 1.5) * np.polyval(rng.normal(0, 1, 4), t).clip(-3, 3) * 0.3 + rng.uniform(0, 1)
    # peak-height variation and detector noise
    y = y * (1 + 0.08 * gaussian_filter1d(rng.normal(0, 1, grid.size), 40) * 6)
    y = y + rng.normal(0, rng.uniform(0.002, 0.03), grid.size)
    return y
