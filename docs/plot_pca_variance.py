"""
Generate the PCA explained-variance figure for the P8 README.

The three measured points come directly from the project notebook:
    50 components  -> 60% variance
    400 components -> 92% variance   (the chosen trade-off)
    600 components -> 95% variance

Run:
    python docs/plot_pca_variance.py
Output:
    docs/pca_variance.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Measured points (from the notebook)
components = [50, 400, 600]
variance = [60, 92, 95]

CHOSEN = 400
CHOSEN_VAR = 92

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)

# Curve through the measured points
ax.plot(components, variance, "-o", color="#1d6fe8", linewidth=2.2,
        markersize=7, label="Explained variance")

# Highlight the chosen trade-off
ax.scatter([CHOSEN], [CHOSEN_VAR], s=180, facecolor="#ff5c8a",
           edgecolor="white", zorder=5, label=f"Chosen: {CHOSEN} comps")
ax.annotate("400 components\n92% variance  (-69% dims)",
            xy=(CHOSEN, CHOSEN_VAR), xytext=(430, 74),
            fontsize=10, fontweight="bold", color="#333",
            arrowprops=dict(arrowstyle="->", color="#ff5c8a", lw=1.6))

# Annotate the other two measured points
ax.annotate("50 → 60%", xy=(50, 60), xytext=(60, 50), fontsize=9, color="#666")
ax.annotate("600 → 95%", xy=(600, 95), xytext=(500, 99), fontsize=9, color="#666")

ax.set_xlabel("Number of PCA components", fontsize=11)
ax.set_ylabel("Explained variance (%)", fontsize=11)
ax.set_title("PCA — explained variance vs. number of components\n"
             "MobileNetV2 features (1280-dim), Fruits-360",
             fontsize=12, fontweight="bold")
ax.set_ylim(40, 102)
ax.set_xlim(0, 680)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
out = __file__.rsplit("/", 1)[0] + "/pca_variance.png"
fig.savefig(out, bbox_inches="tight")
print("Saved:", out)
