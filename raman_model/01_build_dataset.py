"""Collect every named public spectrum (RRUFF excluded) onto one wavenumber grid.

Output: data/unified.npz  (X raw resampled, y class key, name, source, instrument)
Needs: spc-spectra (for .spc files).
"""
import contextlib
import csv
import io
import os
import re
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from raman_prep import GRID, resample

HERE = Path(__file__).resolve().parent
PUB = HERE.parent / "public_raman_data"
csv.field_size_limit(10**9)

rows = []  # (resampled y, display name, source, instrument)
skipped = Counter()


def add(x, y, name, source, instrument):
    r = resample(x, y)
    if r is None or not np.isfinite(r).all() or np.ptp(r) == 0:
        skipped[source] += 1
        return
    rows.append((r, name.strip(), source, instrument))


def key(name):
    k = name.lower().replace("_", " ")
    k = re.sub(r"\s*-\s*", "-", k)
    k = re.sub(r"\s*,\s*", ",", k)
    return re.sub(r"\s+", " ", k).strip()


def read_spc(data):
    import spc_spectra as spc
    fd, p = tempfile.mkstemp(suffix=".spc")
    os.write(fd, data)
    os.close(fd)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            f = spc.File(p)
        x = f.x if getattr(f, "x", None) is not None and len(f.x) else f.sub[0].x
        return np.asarray(x, float), np.asarray(f.sub[0].y, float)
    finally:
        os.remove(p)


# 1. Solvents and reagents (Figshare): one CSV, wavenumber columns + label
f = PUB / "01_solvents_pharma_chemicals/api_compounds_figshare/raman_spectra_api_compounds.csv"
with open(f, encoding="utf-8") as fh:
    rd = csv.reader(fh)
    head = next(rd)
    x = np.array(head[:-1], float)
    for r in rd:
        add(x, np.array(r[:-1], float), r[-1], "solvents_figshare", "Kaiser Rxn2 785nm")

# 2. Pharmaceutical excipients (Zenodo): same compounds on two instruments
for zname, inst in [("FT-Raman_spc_file_format.zip", "FT-Raman 1064nm"), ("LF-785_spc_file_format.zip", "LF 785nm")]:
    zf = zipfile.ZipFile(PUB / "02_pharma_excipients/zenodo_3614035" / zname)
    for n in zf.namelist():
        if not n.lower().endswith(".spc"):
            continue
        stem = Path(n).stem
        stem = re.sub(r"_(FT-Raman|LF-785)$", "", stem)
        parts = re.sub(r"^\d+_", "", stem).split("_")
        name = " ".join(parts[:-1]) if len(parts) > 1 else parts[0]  # last token is the vendor
        try:
            add(*read_spc(zf.read(n)), name, "excipients_zenodo", inst)
        except Exception as e:
            skipped["excipients_zenodo(read error)"] += 1

# 3. Drug adulterants (Mendeley): FT-Raman (.0, plain text) kept; SERS (.spc) is a different technique, left out
for f in sorted((PUB / "06_drug_adulterants/illicit_adulterants_mendeley_y4md8znppn").glob("*.0")):
    try:
        a = np.loadtxt(f, delimiter=",")
        add(a[:, 0], a[:, 1], f.stem, "adulterants_mendeley", "FT-Raman 1064nm")
    except Exception as e:
        skipped["adulterants_mendeley(read error)"] += 1

# 4. Biomolecule reference library
d = PUB / "07_biomolecules_pigments_dyes/biomolecules_reference"
x = np.loadtxt(d / "wnums.csv", delimiter=",")
labels = [r[1] for r in csv.reader(open(d / "labels.csv", encoding="utf-8"))][1:]
for yrow, name in zip(np.loadtxt(d / "spec.csv", delimiter=","), labels):
    add(x, yrow, name, "biomolecules", "mixed (library)")

# 5. Raman Open Database: only entries that cover the whole grid
zf = zipfile.ZipFile(PUB / "03_minerals_inorganics/raman_open_database/rod_all_entries.zip")
num = re.compile(r"_raman_spectrum\.intensity\s*\n((?:\s*-?[\d.]+(?:[eE][-+]?\d+)?\s+-?[\d.]+(?:[eE][-+]?\d+)?\s*\n)+)")
for n in zf.namelist():
    t = zf.read(n).decode("utf-8", "replace")
    m = num.search(t)
    nm = None
    for tag in ("mineral", "common", "systematic"):
        g = re.search(rf"_chemical_name_{tag}\s+(\S.*)", t)
        if g and g.group(1).strip(" '\";?."):
            nm = g.group(1).strip(" '\"")
            break
    if not m or not nm:
        skipped["rod(no data or name)"] += 1
        continue
    a = np.array(m.group(1).split(), float).reshape(-1, 2)
    laser = re.search(r"excitation_laser_wavelength\s+(\S+)", t)
    add(a[:, 0], a[:, 1], nm, "rod", f"ROD {laser.group(1) if laser else '?'}nm")

# Same substance under different names, grades or vendors -> one class (found in the EDA
# as class pairs with near-identical spectra).
ALIASES = [
    (r"^4-methyl-2-pentanone$", "methyl isobutyl ketone"),
    (r"^(plasdone|polyvinylpyrrolidone)\b.*", "polyvinylpyrrolidone"),
    (r"^poly vinyl alcohol\b.*", "polyvinyl alcohol"),
    (r"^polyethylene (glycol|oxide)\b.*", "polyethylene glycol"),
    (r"^magnesium carbonate\b.*", "magnesium carbonate"),
    (r"^carboxymethyl cellulose sodium salt\b.*", "carboxymethyl cellulose sodium"),
    (r"^gelatin\b.*", "gelatin"),
    (r"^(calcium carbonate calcite|calcite)$", "calcium carbonate (calcite)"),
    (r"^(titanium dioxide anatase|anatase)$", "titanium dioxide (anatase)"),
    (r"^(sodium bicarbonate|nahcolite)$", "sodium bicarbonate"),
    (r"^saltpeter$", "potassium nitrate"),
    (r"^nitratine$", "sodium nitrate"),
]


def canonical(k):
    for pattern, target in ALIASES:
        if re.match(pattern, k):
            return target
    return k


X = np.array([r[0] for r in rows])
name = np.array([r[1] for r in rows])
y = np.array([canonical(key(n)) for n in name])
# drop exact repeats of the same spectrum
_, first = np.unique(np.round(X / np.abs(X).max(1, keepdims=True), 6), axis=0, return_index=True)
keep = np.sort(first)
print("exact duplicate spectra removed:", len(X) - len(keep))
rows = [rows[i] for i in keep]
X, name, y = X[keep], name[keep], y[keep]
source = np.array([r[2] for r in rows])
inst = np.array([r[3] for r in rows])
(HERE / "data").mkdir(exist_ok=True)
np.savez_compressed(HERE / "data/unified.npz", X=X, y=y, name=name, source=source, instrument=inst, grid=GRID)

print("spectra", X.shape, "classes", len(set(y)))
for s in sorted(set(source)):
    m = source == s
    print(f"  {s:22s} spectra={m.sum():5d} classes={len(set(y[m])):4d}")
print("skipped (no grid coverage / unreadable):", dict(skipped))
multi = [k for k in set(y) if len(set(source[y == k])) > 1]
print("classes present in more than one source:", sorted(multi))
