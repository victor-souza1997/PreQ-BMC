import numpy as np
import matplotlib.pyplot as plt

# IEEE-friendly style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

backends = [
    "Float32",
    "QNN certified\n(int64)",
    "QNN compact\n(int16)"
]

accuracy = [90.36, 90.24, 90.24]
median_latency = [21.99, 31.68, 27.94]
p95_latency = [22.13, 31.85, 28.08]
rss = [1856, 3132, 1260]

x = np.arange(len(backends))
colors = ["#4C78A8", "#E45756", "#72B7B2"]

# One column, top to bottom; panels share the x axis so labels appear once
fig, axes = plt.subplots(
    3, 1,
    figsize=(3.5, 3.4),
    sharex=True,
    # RSS differs by up to 2.5x, so its panel reads well at a lower height
    gridspec_kw={"height_ratios": [1, 1.4, 0.7]}
)

# ---------------------------------------------------------
# (a) Accuracy
# ---------------------------------------------------------
ax = axes[0]

bars = ax.bar(
    x,
    accuracy,
    color=colors,
    edgecolor="black",
    linewidth=0.6
)

ax.set_ylabel("Acc. (%)")
ax.set_xticks(x)
ax.set_xticklabels(backends)
ax.set_ylim(89.5, 90.6)
ax.set_yticks([89.5, 90.0, 90.5])
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

for bar, value in zip(bars, accuracy):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.03,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=7
    )

# Subfigure caption above the panel
ax.set_title("(a) Classification accuracy", loc="left", fontsize=8, pad=3)

# ---------------------------------------------------------
# (b) Latency
# ---------------------------------------------------------
ax = axes[1]

width = 0.34

bars1 = ax.bar(
    x - width / 2,
    median_latency,
    width,
    label="Median",
    color="#4C78A8",
    edgecolor="black",
    linewidth=0.6
)

bars2 = ax.bar(
    x + width / 2,
    p95_latency,
    width,
    label="P95",
    color="#F2CF5B",
    edgecolor="black",
    linewidth=0.6,
    hatch="//"
)

ax.set_ylabel("Latency (ms)")
ax.set_xticks(x)
ax.set_xticklabels(backends)
# Axis starts at 18 ms (like panel a) so the backend gaps are visible
ax.set_ylim(18, 34)
ax.set_yticks([20, 25, 30])
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

# Legend in one row on the title line, right-aligned, so it never covers bars
ax.legend(
    frameon=False,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.98),
    borderaxespad=0,
    ncol=2,
    handlelength=1.2,
    columnspacing=0.8
)

for bars in (bars1, bars2):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.5
        )

# Subfigure caption above the panel
ax.set_title("(b) Inference latency", loc="left", fontsize=8, pad=3)

# ---------------------------------------------------------
# (c) Peak RSS
# ---------------------------------------------------------
ax = axes[2]

bars = ax.bar(
    x,
    rss,
    color=colors,
    edgecolor="black",
    linewidth=0.6
)

ax.set_ylabel("RSS (KiB)")
ax.set_xticks(x)
ax.set_xticklabels(backends)
ax.set_ylim(0, 3900)
ax.set_yticks([0, 1500, 3000])
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

for bar, value in zip(bars, rss):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 60,
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=7
    )

# Subfigure caption above the panel
ax.set_title("(c) Memory usage", loc="left", fontsize=8, pad=3)

# Remove unnecessary borders
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.align_ylabels(axes)

# Tight vertical spacing: only the panel titles sit between panels
plt.subplots_adjust(
    hspace=0.42,
    bottom=0.12,
    top=0.94,
    left=0.17,
    right=0.99
)

# Final PDF
plt.savefig(
    "vim3_deployment_results.pdf",
    format="pdf",
    bbox_inches="tight"
)

# Optional PNG
plt.savefig(
    "vim3_deployment_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
