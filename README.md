# Raman chemical identification

A model that reads a Raman spectrum and names the chemical. It is trained on public Raman data (360 chemicals) and tested on spectra from our own instrument.

- Full write-up: [raman_model/REPORT.md](raman_model/REPORT.md)
- Slides: [raman_model/Raman_model_report.pptx](raman_model/Raman_model_report.pptx)

## Results in short

| Test | Spectra | Correct first guess |
|---|---|---|
| Public spectra held back from training | 862 | 93.4% |
| Reference and test from different instruments | 75 + 75 | 88.0% and 85.3% |
| Our instrument, the 6 chemicals the model knows | 71 | 82% (58 correct) |

The model only knows chemicals that have public reference spectra. Of the 26 chemicals in our dataset, 6 are covered. Explosives and narcotics have no public Raman data, so the model cannot name them and will give a wrong name instead.

## What is in this repository

| Path | Contents |
|---|---|
| `chem_log.csv` | Our labels: time, chemical name (some rows have a third ON/OFF column) |
| `spec_log.csv` | Our spectra, one per row, 3,648 points. This is the file the model works on |
| `wnums_log.csv` | Wavenumber axis for each row of `spec_log.csv` |
| `spec_wl.csv`, `spec_wol.csv`, `peaks_pxl.csv`, `peaks_amp.csv` | Other files from our instrument. The model does not work on `spec_wl.csv` or `spec_wol.csv` |
| `raman_model/predict.py` | Predict the chemical for chosen rows or a single spectrum |
| `raman_model/04_test_on_my_data.py` | Test the model on the whole dataset |
| `raman_model/01_build_dataset.py`, `02_eda.py`, `03_train.py` | Build the training set, analyse it, train the model |
| `raman_model/raman_prep.py` | Preprocessing shared by training and prediction |
| `raman_model/model/raman_model.joblib` | The trained model (69 MB) |
| `raman_model/data/unified.npz` | The combined public training spectra (4,120 spectra, 360 chemicals) |
| `raman_model/eda/`, `raman_model/results/` | Figures, metrics, chemical lists and test results |

## Setup

Python 3.10 or newer.

```
git clone https://github.com/ritik3041998/raman_new.git
cd raman_new
pip install -r requirements.txt
```

## How to run the model

All commands are run from the `raman_model` folder:

```
cd raman_model
```

Row numbers are the line numbers shown in `chem_log.csv`, starting at 1.

**Predict one spectrum from the dataset**

```
python predict.py --spec ..\spec_log.csv --wnums ..\wnums_log.csv --labels ..\chem_log.csv --row 700
```

Output:

```
Row 700  true: Acetone              predicted: Acetone                          MATCH
```

**Predict several rows or a range**

```
python predict.py --spec ..\spec_log.csv --wnums ..\wnums_log.csv --labels ..\chem_log.csv --row 13 265 700-701
```

**Predict every row and save the result**

```
python predict.py --spec ..\spec_log.csv --wnums ..\wnums_log.csv --labels ..\chem_log.csv --all --out predictions.csv
```

**Predict a new spectrum from its own file**

The file needs two columns, wavenumber and intensity, one point per line. An example is in `results/example_single_spectrum.csv`.

```
python predict.py --file results\example_single_spectrum.csv
```

**Test on the chemicals the model knows**

The rows of our dataset whose chemical is in the training set are listed in `results/matching_chemicals.csv`. For example, all sodium nitrate rows:

```
python predict.py --spec ..\spec_log.csv --wnums ..\wnums_log.csv --labels ..\chem_log.csv --row 265 266 267 268 269 270 271 272 273 274 275 298 299 301 302 622 623 624 628 629
```

**Test on the whole dataset with a per-chemical summary**

```
python 04_test_on_my_data.py spec_log.csv
```

Notes:

- `--labels` is optional. It only prints the true name next to the prediction.
- A spectrum must cover 450 to 1,800 cm-1.
- On macOS or Linux, write the paths with `/` instead of `\`.
- The first run takes a few seconds while the model loads.

## Retrain the model

```
python 02_eda.py
python 03_train.py
```

`02_eda.py` writes the figures and summary to `eda/`. `03_train.py` compares preprocessing options, models and feature sets, then saves the final model to `model/raman_model.joblib`. It takes about 20 minutes.

`01_build_dataset.py` rebuilds `data/unified.npz` from the raw public downloads. Those downloads (about 1.4 GB) are not in this repository; the script expects them in a `public_raman_data` folder next to `raman_model`, and also needs `pip install spc-spectra`.

## How the model works

1. Each spectrum is resampled to 450 to 1,800 cm-1, smoothed, has its background removed, is square-root scaled and normalised.
2. The 700 most useful wavenumbers (of 1,351) are kept.
3. Two methods are combined: matching against a library of reference spectra, and a logistic regression classifier.

## Training data sources

| Source | Chemicals | Licence |
|---|---|---|
| Solvents and reagents, Figshare doi:10.6084/m9.figshare.27931131 | 31 | CC BY |
| Pharmaceutical excipients, Zenodo record 3614035 | 56 | CC BY 4.0 |
| Raman Open Database (solsa.crystallography.net/rod) | 127 | CC0 |
| Biomolecule reference library (ramanbiolib) | 141 | see source |
| Drug adulterants, Mendeley Data y4md8znppn | 11 | see source |

The biomolecule and adulterant sets were obtained through the `raman-data` package (github.com/ml-lab-htw/raman_data).

## Known limitations

- Covers 6 of our 26 chemicals and none of the explosives.
- Always returns a name, even for a chemical it was not trained on.
- Uses only 450 to 1,800 cm-1.
- Most training chemicals have one or two reference spectra, so their real-world accuracy is unmeasured.
