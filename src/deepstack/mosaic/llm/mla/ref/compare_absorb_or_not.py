# illustration https://zhuanlan.zhihu.com/p/1941992212989732340
import numpy as np
import matplotlib.pyplot as plt

def y_boundary(x):
    # y = 3x^2 / (512 - 3x) is meaningful only when x < 512/3 and y>=0
    return (3.0 * x**2) / (512.0 - 3.0 * x)

# ---- Main parameters ----
x_max_plot = 512
x_asymptote = 512.0 / 3.0  # ≈170.6667

# Display the full x-axis (0~8192), but draw the curve and fill only where x < the asymptote
x = np.linspace(0.0, x_max_plot, 5000)

# Keep only the valid portion where x < the asymptote for the curve/fill
mask_valid = x < x_asymptote
x_valid = x[mask_valid]

# Boundary curve (use NaN in invalid regions to prevent incorrect fill_between shading)
y_b_valid = y_boundary(x_valid)

# Choose a reasonable y upper limit for visualization (avoid divergence near the asymptote)
# Use the y value at 99% of the asymptote's x-coordinate as a reference upper limit
x_ref = x_asymptote * 0.97
y_ref = y_boundary(x_ref)
y_top = float(y_ref) * 1.05 if np.isfinite(y_ref) and y_ref > 0 else 1e6

# ---- Plotting ----
# Use a consistent font size
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

plt.figure(figsize=(6, 4))
ax = plt.gca()

# 1) Boundary curve (draw only the valid region)
# plt.plot(x_valid, y_b_valid, label=r"Boundary: $512y = 3x^2 + 3xy$")
plt.plot(x_valid, y_b_valid, label=r"Boundary")

# 2) Within the valid region, shade only the feasible area: y > y_boundary(x)
y_fill_top = np.full_like(x_valid, y_top)
plt.fill_between(x_valid, y_b_valid, y_fill_top,
                 where=(y_fill_top > y_b_valid),
                 alpha=0.2, label="MLA absorbs better")

# 3) Annotate the infeasible region where x >= the asymptote
# plt.axvline(x_asymptote, linestyle="--")
# plt.text(x_asymptote * 1.01, y_top * 0.85,
#          r"$x=\frac{512}{3}$ Asymptote",
#          rotation=90, va="top")
# plt.text(x_asymptote + (x_max_plot - x_asymptote) * 0.25,
#          y_top * 0.15,
#          "No feasible region for inequality\nwhen x ≥ 512/3 (with y ≥ 0)",
#          ha="center")

# 4) Other plot elements
plt.xlim(0, x_max_plot)
plt.ylim(0, y_top)
plt.xlabel("Sequence Length", fontsize=LABEL_FONTSIZE)
plt.ylabel("KV Cache Length", fontsize=LABEL_FONTSIZE)
# plt.title("512y > 3x^2 + 3xy Region (x, y ≥ 0, matmul-only)")
plt.title("Divisions in MLA: Absorb or not Matrices", fontsize=TITLE_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE)
plt.grid(True)
plt.tight_layout()

# Save and display
plt.savefig("compare_absorb_or_not.png", dpi=300)
