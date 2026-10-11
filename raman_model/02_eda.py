"""Exploratory analysis of the unified dataset. Writes figures to eda/ and eda/summary.txt.

Also caches the preprocessed spectra in data/preprocessed.npy for the training script.
"""
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from plotstyle import INK2, MUTED, SERIES, SOURCE_COLOR, SOURCE_LABEL, SOURCES, plt
from raman_prep import GRID, preprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / "eda"
OUT.mkdir(exist_ok=True)
d = np.load(HERE / "data/unified.npz")
X, y, name, source = d["X"], d["y"], d["name"], d["source"]
lines = []


def say(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    lines.append(s)


Xp = preprocess(X)  # baseline removal + L2, the default pipeline
np.save(HERE / "data/preprocessed.npy", Xp)

# ---- 1. size of each source
counts = Counter(y)
say(f"Spectra: {len(y)}   classes: {len(counts)}   grid: {GRID[0]:.0f}-{GRID[-1]:.0f} cm-1, {GRID.size} points")
fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
for ax, vals, title in [
    (axes[0], [np.sum(source == s) for s in SOURCES], "Spectra per source"),
    (axes[1], [len(set(y[source == s])) for s in SOURCES], "Chemical classes per source"),
]:
    labels = [SOURCE_LABEL[s] for s in SOURCES]
    bars = ax.barh(labels[::-1], vals[::-1], color=[SOURCE_COLOR[s] for s in SOURCES][::-1], height=0.6)
    ax.bar_label(bars, padding=3, color=INK2, fontsize=9)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.15)
axes[1].set_yticklabels([])
fig.tight_layout()
fig.savefig(OUT / "01_sources.png")
plt.close(fig)
for s in SOURCES:
    say(f"  {SOURCE_LABEL[s]:26s} spectra={np.sum(source == s):5d} classes={len(set(y[source == s])):4d}")

# ---- 2. how many real spectra each class has
v = np.array(list(counts.values()))
bins = [("1", v == 1), ("2", v == 2), ("3-5", (v >= 3) & (v <= 5)), ("6-50", (v >= 6) & (v <= 50)), (">50", v > 50)]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
bars = ax.bar([b[0] for b in bins], [int(b[1].sum()) for b in bins], color=SERIES[0], width=0.6)
ax.bar_label(bars, padding=2, color=INK2, fontsize=9)
ax.set_xlabel("Real spectra available for the class")
ax.set_ylabel("Number of classes")
ax.set_title("Most classes have only one or two spectra")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "02_spectra_per_class.png")
plt.close(fig)
say("Classes by number of real spectra:", ", ".join(f"{b[0]}: {int(b[1].sum())}" for b in bins))

# ---- 3. effect of preprocessing on one spectrum per source
fig, axes = plt.subplots(2, len(SOURCES), figsize=(13, 4.4), sharex=True)
for j, s in enumerate(SOURCES):
    i = np.flatnonzero(source == s)[0]
    axes[0, j].plot(GRID, X[i], color=SOURCE_COLOR[s])
    axes[1, j].plot(GRID, Xp[i], color=SOURCE_COLOR[s])
    axes[0, j].set_title(f"{SOURCE_LABEL[s]}\n{name[i][:26]}", fontsize=9)
    axes[1, j].set_xlabel("Raman shift (cm$^{-1}$)")
    for a in axes[:, j]:
        a.set_yticks([])
axes[0, 0].set_ylabel("Raw")
axes[1, 0].set_ylabel("Preprocessed")
fig.suptitle("Raw spectra differ in scale and background; preprocessing puts them on one footing",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "03_raw_vs_preprocessed.png")
plt.close(fig)
for s in SOURCES:
    m = source == s
    say(f"  raw intensity range {SOURCE_LABEL[s]:26s} min={X[m].min():.3g} median={np.median(X[m]):.3g} max={X[m].max():.3g}")

# ---- 4. where the signal sits: mean preprocessed spectrum per source
fig, axes = plt.subplots(len(SOURCES), 1, figsize=(8.5, 6.6), sharex=True)
for ax, s in zip(axes, SOURCES):
    cls_means = np.array([Xp[(source == s) & (y == c)].mean(0) for c in sorted(set(y[source == s]))])
    ax.fill_between(GRID, cls_means.mean(0), color=SOURCE_COLOR[s], alpha=0.25, linewidth=0)
    ax.plot(GRID, cls_means.mean(0), color=SOURCE_COLOR[s])
    ax.set_yticks([])
    ax.text(0.995, 0.82, SOURCE_LABEL[s], transform=ax.transAxes, ha="right", color=INK2, fontsize=9)
axes[0].set_title("Average spectrum of each source (mean over its classes)")
axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
fig.tight_layout()
fig.savefig(OUT / "04_mean_spectrum_per_source.png")
plt.close(fig)

# ---- 5. PCA: do sources separate (instrument effect) or mix (chemistry)?
pca = PCA(10, random_state=0).fit(Xp)
Z = pca.transform(Xp)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for s in SOURCES:
    m = source == s
    ax.scatter(Z[m, 0], Z[m, 1], s=14, color=SOURCE_COLOR[s], alpha=0.55, linewidth=0, label=SOURCE_LABEL[s])
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("PCA of preprocessed spectra, coloured by source")
ax.legend(loc="best", markerscale=1.6)
fig.tight_layout()
fig.savefig(OUT / "05_pca_by_source.png")
plt.close(fig)
full = PCA(random_state=0).fit(Xp)
cum = np.cumsum(full.explained_variance_ratio_)
say("PCA components needed for 90 / 95 / 99 % variance:",
    int(np.searchsorted(cum, 0.90) + 1), int(np.searchsorted(cum, 0.95) + 1), int(np.searchsorted(cum, 0.99) + 1))

# ---- 6. classes that look alike (will be confused whatever the model)
classes = np.array(sorted(counts))
means = np.array([Xp[y == c].mean(0) for c in classes])
means /= np.linalg.norm(means, axis=1, keepdims=True)
sim = means @ means.T
np.fill_diagonal(sim, 0)
iu = np.triu_indices_from(sim, 1)
order = np.argsort(sim[iu])[::-1]
say(f"Class pairs with cosine similarity > 0.98: {int((sim[iu] > 0.98).sum())}   > 0.95: {int((sim[iu] > 0.95).sum())}")
say("Most similar class pairs:")
for k in order[:25]:
    say(f"  {sim[iu][k]:.3f}  {classes[iu[0][k]]}  <->  {classes[iu[1][k]]}")
nn = sim.max(1)
fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.hist(nn, bins=np.linspace(0, 1, 41), color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.8)
ax.axvline(0.95, color=MUTED, linewidth=1, linestyle="--")
ax.text(0.945, ax.get_ylim()[1] * 0.92, "hard to tell apart", ha="right", color=INK2, fontsize=9)
ax.set_xlabel("Cosine similarity to the most similar other class")
ax.set_ylabel("Number of classes")
ax.set_title("How distinct each class is from its nearest neighbour")
fig.tight_layout()
fig.savefig(OUT / "06_nearest_class_similarity.png")
plt.close(fig)

# ---- 7. data quality
snr = Xp.max(1) / (np.std(np.diff(Xp, axis=1), axis=1) + 1e-12)
for s in SOURCES:
    say(f"  peak-to-noise (median) {SOURCE_LABEL[s]:26s} {np.median(snr[source == s]):8.1f}")
dup = int((np.unique(np.round(Xp, 4), axis=0).shape[0] != Xp.shape[0]))
say("Exact duplicate spectra present:", "yes" if dup else "no")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
