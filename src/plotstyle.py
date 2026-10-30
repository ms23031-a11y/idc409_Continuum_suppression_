import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
POS, NEG = "#0072B2", "#D55E00"      # colour-blind-safe blue / orange
INK, MUTED = "#222222", "#6b6b6b"
TYPE_COL = {0: "#009E73", 1: "#0072B2", 2: "#D55E00", 3: "#CC79A7", 4: "#E69F00", 5: "#56B4E9"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": .6,
                     "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "savefig.bbox": "tight"})
