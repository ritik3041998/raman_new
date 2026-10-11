"""Predict the chemical name for individual spectra.

Three ways to give a spectrum:

  1. One row of our dataset files (row = line number shown in the editor, starting at 1):
       python predict.py --spec ../spec_log.csv --wnums ../wnums_log.csv --row 12
     several rows, or a range:
       python predict.py --spec ../spec_log.csv --wnums ../wnums_log.csv --row 12 40 100-105
     add --labels ../chem_log.csv to print the true name next to the prediction.

  2. A single two-column file (wavenumber, intensity), one point per line:
       python predict.py --file my_spectrum.csv

  3. Every row of a dataset:
       python predict.py --spec ../spec_log.csv --wnums ../wnums_log.csv --all --out predictions.csv

The spectrum must cover 450-1800 cm-1.
"""
import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

from raman_prep import preprocess, resample

HERE = Path(__file__).resolve().parent

# model name -> name as written in chem_log.csv, for the chemicals both have
CHEM_LOG_NAME = {
    "acetone": "Acetone",
    "ethanol": "Ethyl Alcohol",
    "2-propanol": "Isopropyl Alcohol",
    "potassium nitrate": "Potassium Nitrate",
    "sodium nitrate": "Sodium Nitrate",
    "potassium carbonate": "Potassium Carbonate",
}


def read_rows(path, wanted=None):
    """Rows of a comma-separated numeric file; only the wanted row numbers if given."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if wanted is None or i + 1 in wanted:
                out[i + 1] = np.array(line.strip().split(","), float)
    return out


def parse_rows(items):
    rows = []
    for it in items:
        if "-" in it:
            a, b = it.split("-")
            rows += list(range(int(a), int(b) + 1))
        else:
            rows.append(int(it))
    return rows


ap = argparse.ArgumentParser(description="Predict chemical names from Raman spectra.")
ap.add_argument("--spec", help="intensity file, one spectrum per row")
ap.add_argument("--wnums", help="wavenumber file: one row per spectrum, or a single row used for all")
ap.add_argument("--row", nargs="+", help="line numbers (from 1, as in the editor) or ranges like 10-15")
ap.add_argument("--all", action="store_true", help="predict every row of --spec")
ap.add_argument("--labels", help="optional label file (time, name per row) to show the true name")
ap.add_argument("--file", help="two-column file: wavenumber, intensity")
ap.add_argument("--out", help="also save the predictions to this CSV file")
a = ap.parse_args()

# ---- collect the spectra to predict: list of (id, wavenumbers, intensities)
items = []
if a.file:
    t = np.genfromtxt(a.file, delimiter=",")
    if t.ndim != 2 or t.shape[1] < 2:
        t = np.genfromtxt(a.file)  # whitespace or tab separated
    t = t[np.isfinite(t[:, :2]).all(1)]
    items.append((Path(a.file).name, t[:, 0], t[:, 1]))
elif a.spec and a.wnums and (a.row or a.all):
    wanted = None if a.all else set(parse_rows(a.row))
    spec = read_rows(a.spec, wanted)
    wn = read_rows(a.wnums, wanted)
    if not wn:
        wn = read_rows(a.wnums, {1})  # single shared axis
    missing = sorted(set(wanted or []) - set(spec))
    if missing:
        sys.exit(f"Row(s) {missing} not found: {a.spec} has rows 1 to {max(read_rows(a.spec))}.")
    for i in sorted(spec):
        items.append((i, wn.get(i, next(iter(wn.values()))), spec[i]))
else:
    ap.print_help()
    sys.exit(1)

labels = {}
if a.labels:
    # name is the second column; some rows carry extra columns after it
    labels = {i: r[1].strip() for i, r in enumerate(csv.reader(open(a.labels, encoding="utf-8")), 1) if len(r) > 1}

# ---- predict
b = joblib.load(HERE / "model/raman_model.joblib")
classes, model = b["classes"], b["model"]
results = []
for ident, w, y in items:
    if len(w) != len(y):
        results.append((ident, None, f"wavenumber and intensity lengths differ ({len(w)} vs {len(y)})"))
        continue
    r = resample(w, y)
    if r is None or np.ptp(r) == 0:
        results.append((ident, None, f"spectrum does not cover 450-1800 cm-1 (has {np.min(w):.0f} to {np.max(w):.0f}) or is flat"))
        continue
    F = preprocess(r[None], mask=b["mask"], **b["variant"])
    name = str(classes[model.scores(F)[0].argmax()])
    results.append((ident, CHEM_LOG_NAME.get(name, name), None))

# ---- report
for ident, name, err in results:
    if name is None:
        print(f"Row {ident}: cannot predict: {err}")
    elif ident in labels:
        print(f"Row {ident:<5}true: {labels[ident]:20s} predicted: {name:32s} {'MATCH' if name == labels[ident] else 'no match'}")
    else:
        print(f"Row {ident}   predicted: {name}")

judged = [(labels[i], n) for i, n, _ in results if i in labels and n is not None]
if len(judged) > 1:
    print()
    print(f"matched {sum(t == n for t, n in judged)} of {len(judged)}")

if a.out:
    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["spectrum", "true_label", "prediction", "match"])
        for ident, name, err in results:
            hit = ("yes" if name == labels[ident] else "no") if ident in labels and name is not None else ""
            wr.writerow([ident, labels.get(ident, ""), name if name is not None else err, hit])
    print("saved", a.out)
