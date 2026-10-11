"""Shared matplotlib style for the EDA and training figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, SURFACE, GRID_C = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "legend.frameon": False, "lines.linewidth": 1.4, "font.family": "DejaVu Sans", "figure.dpi": 130,
})

# Fixed colour per data source, so a source keeps its colour in every figure.
SOURCES = ["solvents_figshare", "excipients_zenodo", "biomolecules", "rod", "adulterants_mendeley"]
SOURCE_LABEL = {
    "solvents_figshare": "Solvents", "excipients_zenodo": "Excipients", "biomolecules": "Biomolecules",
    "rod": "Inorganic crystals (ROD)", "adulterants_mendeley": "Drug adulterants",
}
SOURCE_COLOR = dict(zip(SOURCES, SERIES))
