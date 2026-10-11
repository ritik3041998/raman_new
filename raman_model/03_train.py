"""Choose preprocessing, model and features; train the final chemical-name model.

Evaluation uses spectra the model has not seen:
  holdout     real spectra held back from every class that has at least two
  FT->LF      excipients: reference from the FT-Raman instrument, test on the 785 nm instrument
  LF->FT      the reverse
  synthetic   augmented copies (the only possible test for classes with one spectrum)
Outputs: model/raman_model.joblib, results/*.json|csv|png
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from plotstyle import INK2, MUTED, SERIES, plt
from raman_prep import GRID, CosineMatcher, Ensemble, augment, finish, remove_baseline

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
(HERE / "model").mkdir(exist_ok=True)
d = np.load(HERE / "data/unified.npz")
X, y, source, inst = d["X"], d["y"], d["source"], d["instrument"]
classes = np.unique(y)
T0 = time.time()

VARIANTS = {
    "no baseline removal": dict(baseline=False),
    "baseline": dict(),
    "baseline + sqrt": dict(sqrt=True),
    "baseline + derivative": dict(derivative=True),
}
PER_CLASS, REAL_CAP = 60, 30


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def build_train(train_idx, seed, per_class=PER_CLASS):
    """Raw training matrix: up to REAL_CAP real spectra per class, topped up with augmented copies."""
    rng = np.random.default_rng(seed)
    rows, labels, is_real = [], [], []
    for c in classes:
        idx = train_idx[y[train_idx] == c]
        if idx.size == 0:
            continue
        real = rng.choice(idx, min(idx.size, REAL_CAP), replace=False)
        rows += [X[i] for i in real]
        labels += [c] * real.size
        is_real += [True] * real.size
        for _ in range(per_class - real.size):
            rows.append(augment(X[rng.choice(idx)], rng))
            labels.append(c)
            is_real.append(False)
    return np.array(rows), np.array(labels), np.array(is_real)


def stage(raw):
    """(without baseline removal, with baseline removal) so every variant can be derived cheaply."""
    return raw - raw.min(axis=1, keepdims=True), remove_baseline(raw)


def feats(staged, baseline=True, **kw):
    return finish(staged[1] if baseline else staged[0], **kw)


def make_models():
    return {
        "Library match, real spectra only": "match_real",
        "Library match + augmentation": "match",
        "PCA + logistic regression": make_pipeline(PCA(100, random_state=0), StandardScaler(),
                                                    LogisticRegression(C=20, max_iter=400)),
        "PCA + SVM (RBF)": make_pipeline(PCA(100, random_state=0), StandardScaler(),
                                         SVC(C=10, decision_function_shape="ovr")),
        "PCA + neural network": make_pipeline(PCA(100, random_state=0), StandardScaler(),
                                              MLPClassifier((512,), early_stopping=True, max_iter=200, random_state=0)),
        "Extra trees": ExtraTreesClassifier(300, n_jobs=-1, random_state=0),
        "Ensemble: match + logistic": Ensemble(make_pipeline(PCA(100, random_state=0), StandardScaler(),
                                                             LogisticRegression(C=20, max_iter=400))),
    }


def fit_scores(model, Ftr, ytr, real, tests):
    """Fit one model and return class scores for each test matrix."""
    if isinstance(model, str):
        m = CosineMatcher().fit(Ftr[real] if model == "match_real" else Ftr, ytr[real] if model == "match_real" else ytr)
        return m.classes_, [m.scores(F) for F in tests]
    if isinstance(model, Ensemble):
        model.fit(Ftr, ytr)
        return model.classes_, [model.scores(F) for F in tests]
    cls, codes = np.unique(ytr, return_inverse=True)  # integer labels: MLP early stopping needs them
    model.fit(Ftr, codes)
    fn = model.decision_function if hasattr(model, "decision_function") else model.predict_proba
    return cls, [fn(F) for F in tests]


def topk(cls, S, ytrue, k):
    order = np.argsort(-S, axis=1)[:, :k]
    return float(np.mean([t in cls[o] for t, o in zip(ytrue, order)]))


# ---------------------------------------------------------------- evaluation splits
rng = np.random.default_rng(0)
hold = []
for c in classes:
    idx = np.flatnonzero(y == c)
    if idx.size >= 2:
        hold += list(rng.choice(idx, 1 if idx.size <= 5 else int(round(0.2 * idx.size)), replace=False))
hold = np.array(sorted(hold))
exc = source == "excipients_zenodo"
ft, lf = np.flatnonzero(exc & (inst == "FT-Raman 1064nm")), np.flatnonzero(exc & (inst == "LF 785nm"))
allidx = np.arange(len(y))
SPLITS = {
    "holdout": (np.setdiff1d(allidx, hold), hold),
    "FT->LF": (np.setdiff1d(allidx, lf), lf[np.isin(y[lf], y[ft])]),
    "LF->FT": (np.setdiff1d(allidx, ft), ft[np.isin(y[ft], y[lf])]),
}
log("classes", len(classes), "| test sizes", {k: len(v[1]) for k, v in SPLITS.items()})

prepared = {}
for i, (sname, (tr, te)) in enumerate(SPLITS.items()):
    raw, ytr, real = build_train(tr, seed=10 + i)
    entry = dict(train=stage(raw), ytr=ytr, real=real, tests={sname: (stage(X[te]), y[te])})
    if sname == "holdout":
        r2 = np.random.default_rng(99)
        syn_y = np.repeat(np.unique(y[tr]), 5)
        syn = np.array([augment(X[r2.choice(tr[y[tr] == c])], r2) for c in syn_y])
        entry["tests"]["synthetic"] = (stage(syn), syn_y)
    prepared[sname] = entry
    log("prepared split", sname, "train", raw.shape)
TESTS = ["holdout", "FT->LF", "LF->FT", "synthetic"]


def evaluate(model_factory, variant, mask=None):
    """Top-1 / top-5 on every test set for one model + preprocessing variant (+ feature mask)."""
    out = {}
    for sname, e in prepared.items():
        Ftr = feats(e["train"], mask=mask, **variant)
        names = list(e["tests"])
        cls, S = fit_scores(model_factory(), Ftr, e["ytr"], e["real"],
                            [feats(e["tests"][n][0], mask=mask, **variant) for n in names])
        for n, s in zip(names, S):
            out[n] = (topk(cls, s, e["tests"][n][1], 1), topk(cls, s, e["tests"][n][1], 5))
    return out


def select_score(r):
    """Model-selection score: real unseen spectra only, cross-instrument counted as one test."""
    return 0.5 * r["holdout"][0] + 0.25 * (r["FT->LF"][0] + r["LF->FT"][0])


def show(title, table):
    log(title)
    print(f"  {'':36s}" + "".join(f"{t:>16s}" for t in TESTS) + f"{'score':>8s}")
    for k, r in table.items():
        print(f"  {k:36s}" + "".join(f"{r[t][0]:7.1%} /{r[t][1]:6.1%}" for t in TESTS) + f"{select_score(r):8.3f}", flush=True)


results = {}
# ---------------------------------------------------------------- 1. preprocessing
t1 = {}
for vname, v in VARIANTS.items():
    for mname in ["Library match + augmentation", "PCA + logistic regression"]:
        t1[f"{vname} | {mname[:14]}"] = evaluate(lambda m=mname: make_models()[m], v)
show("1. preprocessing variants (top-1 / top-5)", t1)
results["preprocessing"] = t1
vscore = {vn: np.mean([select_score(r) for k, r in t1.items() if k.startswith(vn + " |")]) for vn in VARIANTS}
best_variant = max(vscore, key=vscore.get)
log("best preprocessing:", best_variant)

# ---------------------------------------------------------------- 2. models
t2 = {m: evaluate(lambda m=m: make_models()[m], VARIANTS[best_variant]) for m in make_models()}
show("2. models (top-1 / top-5)", t2)
results["models"] = t2
best_model = max(t2, key=lambda m: select_score(t2[m]))
log("best model:", best_model)

# ---------------------------------------------------------------- 3. important features
e = prepared["holdout"]
Fh = feats(e["train"], **VARIANTS[best_variant])
fscore = np.nan_to_num(f_classif(Fh, e["ytr"])[0])
rank = np.argsort(-fscore)
t3 = {}
for k in [100, 200, 400, 700, 1000, GRID.size]:
    mask = np.zeros(GRID.size, bool)
    mask[rank[:k]] = True
    t3[f"top {k} wavenumbers"] = evaluate(lambda: make_models()[best_model], VARIANTS[best_variant], mask=mask)
show("3. feature selection with the best model (top-1 / top-5)", t3)
results["features"] = t3
top = max(select_score(r) for r in t3.values())
# smallest feature set within half a point of the best
best_k = min(int(k.split()[1]) for k, r in t3.items() if select_score(r) >= top - 0.005)
mask = np.zeros(GRID.size, bool)
mask[rank[:best_k]] = True
log("features kept:", best_k, "of", GRID.size)

# ---------------------------------------------------------------- 4. final model on all data
raw, ytr, real = build_train(allidx, seed=7, per_class=80)
Ftr = finish(remove_baseline(raw), mask=mask, **{k: v for k, v in VARIANTS[best_variant].items() if k != "baseline"}) \
    if VARIANTS[best_variant].get("baseline", True) else feats(stage(raw), mask=mask, **VARIANTS[best_variant])
spec = make_models()[best_model]
if isinstance(spec, str):
    use = real if spec == "match_real" else np.ones(len(ytr), bool)
    final = CosineMatcher().fit(Ftr[use], ytr[use])
    final_classes = final.classes_
elif isinstance(spec, Ensemble):
    final = spec.fit(Ftr, ytr)
    final_classes = final.classes_
else:
    final_classes, codes = np.unique(ytr, return_inverse=True)
    final = spec.fit(Ftr, codes)
joblib.dump(dict(model=final, classes=final_classes, grid=GRID, mask=mask, variant=VARIANTS[best_variant],
                 variant_name=best_variant, model_name=best_model), HERE / "model/raman_model.joblib", compress=3)
log("saved model:", best_model, "| classes", len(final_classes), "| training rows", len(ytr))

# ---------------------------------------------------------------- outputs
results["chosen"] = dict(preprocessing=best_variant, model=best_model, features_kept=best_k,
                         n_classes=len(classes), n_real_spectra=len(y))
(RES / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
np.savetxt(RES / "feature_importance.csv", np.c_[GRID, fscore, mask], delimiter=",",
           header="wavenumber,f_score,kept", comments="", fmt=["%.0f", "%.4f", "%d"])
(RES / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(GRID, fscore, color=MUTED, linewidth=1, label="not kept")
kept = np.where(mask, fscore, np.nan)
ax.plot(GRID, kept, color=SERIES[0], linewidth=1.6, label=f"kept ({best_k} wavenumbers)")
ax.set_xlabel("Raman shift (cm$^{-1}$)")
ax.set_ylabel("ANOVA F-score")
ax.set_title("Which wavenumbers separate the chemical classes")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(RES / "feature_importance.png")
plt.close(fig)

names = list(t2)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, t, title in zip(axes, ["holdout", "FT->LF", "LF->FT"],
                        ["Held-out real spectra", "Other instrument (FT to 785 nm)", "Other instrument (785 nm to FT)"]):
    vals = [t2[m][t][0] * 100 for m in names][::-1]
    bars = ax.barh(names[::-1], vals, color=[SERIES[0] if m == best_model else "#b7d3f6" for m in names][::-1], height=0.6)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, color=INK2, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(RES / "model_comparison.png")
plt.close(fig)
log("done")
