"""Test the trained model on our own instrument data (the CSV files one folder up).

Usage: python 04_test_on_my_data.py [spec_log.csv]
Writes results/my_data_predictions_<file>.csv with the top-5 names for every spectrum.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from raman_prep import preprocess, resample

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
spec_file = sys.argv[1] if len(sys.argv) > 1 else "spec_log.csv"

# our label -> name used by the model (only chemicals the public data contains)
NAME_MAP = {
    "Acetone": "acetone",
    "Ethyl Alcohol": "ethanol",
    "Isopropyl Alcohol": "2-propanol",
    "Potassium Nitrate": "potassium nitrate",
    "Sodium Nitrate": "sodium nitrate",
    "Potassium Carbonate": "potassium carbonate",
}

b = joblib.load(HERE / "model/raman_model.joblib")
classes = b["classes"]
labels = [r[1].strip() for r in csv.reader(open(DATA / "chem_log.csv", encoding="utf-8"))]
spec = np.loadtxt(DATA / spec_file, delimiter=",")
wn = np.loadtxt(DATA / "wnums_log.csv", delimiter=",")

R = [resample(w, s) for w, s in zip(wn, spec)]
ok = np.array([r is not None and np.ptp(r) > 0 for r in R])
F = preprocess(np.array([r for r, k in zip(R, ok) if k]), mask=b["mask"], **b["variant"])
S = b["model"].scores(F)
match = b["model"].matcher.scores(F)  # cosine similarity to the closest reference, 0..1
order = np.argsort(-S, axis=1)[:, :5]

rows = []
j = 0
for i, lab in enumerate(labels):
    if not ok[i]:
        rows.append([i, lab, NAME_MAP.get(lab, ""), "unusable spectrum", "", "", "", "", ""])
        continue
    top = list(classes[order[j]])
    rows.append([i, lab, NAME_MAP.get(lab, ""), *top, f"{match[j].max():.3f}"])
    j += 1
out = HERE / f"results/my_data_predictions_{Path(spec_file).stem}.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows([["row", "true_label", "model_name", "top1", "top2", "top3", "top4", "top5", "best_match_similarity"]] + rows)

print(f"file: {spec_file}   spectra: {len(labels)}   usable: {int(ok.sum())}")
known = [r for r in rows if r[2] and r[3] != "unusable spectrum"]
other = [r for r in rows if not r[2] and r[3] != "unusable spectrum"]
print(f"\nChemicals the model knows: {len(known)} spectra")
per = defaultdict(list)
for r in known:
    per[r[1]].append(r)
print(f"  {'chemical':22s}{'n':>4s}{'top-1':>8s}{'top-5':>8s}  median similarity   most common prediction")
for lab, rs in sorted(per.items(), key=lambda t: -len(t[1])):
    t1 = np.mean([r[3] == r[2] for r in rs])
    t5 = np.mean([r[2] in r[3:8] for r in rs])
    common = Counter(r[3] for r in rs).most_common(1)[0]
    print(f"  {lab:22s}{len(rs):4d}{t1:8.0%}{t5:8.0%}  {np.median([float(r[8]) for r in rs]):8.3f}            {common[0]} ({common[1]})")
if known:
    print(f"  {'ALL':22s}{len(known):4d}{np.mean([r[3] == r[2] for r in known]):8.0%}{np.mean([r[2] in r[3:8] for r in known]):8.0%}")

print(f"\nChemicals the model does not know: {len(other)} spectra (it cannot be right; shows what it says instead)")
per = defaultdict(list)
for r in other:
    per[r[1]].append(r)
for lab, rs in sorted(per.items(), key=lambda t: -len(t[1])):
    common = Counter(r[3] for r in rs).most_common(1)[0]
    print(f"  {lab:22s}{len(rs):4d}  median similarity {np.median([float(r[8]) for r in rs]):.3f}   mostly predicted: {common[0]} ({common[1]})")
print("\npredictions written to", out)
