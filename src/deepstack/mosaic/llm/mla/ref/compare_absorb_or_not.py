# illustration https://zhuanlan.zhihu.com/p/1941992212989732340
import numpy as np
import matplotlib.pyplot as plt

def y_boundary(x):
    return (3.0 * x**2) / (512.0 - 3.0 * x)

x_max_plot = 512
x_asymptote = 512.0 / 3.0  # ≈170.6667

x = np.linspace(0.0, x_max_plot, 5000)

mask_valid = x < x_asymptote
x_valid = x[mask_valid]

y_b_valid = y_boundary(x_valid)

x_ref = x_asymptote * 0.97
y_ref = y_boundary(x_ref)
y_top = float(y_ref) * 1.05 if np.isfinite(y_ref) and y_ref > 0 else 1e6

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

plt.figure(figsize=(6, 4))
ax = plt.gca()

# plt.plot(x_valid, y_b_valid, label=r"Boundary: $512y = 3x^2 + 3xy$")
plt.plot(x_valid, y_b_valid, label=r"Boundary")

y_fill_top = np.full_like(x_valid, y_top)
plt.fill_between(x_valid, y_b_valid, y_fill_top,
                 where=(y_fill_top > y_b_valid),
                 alpha=0.2, label="MLA absorbs better")

# plt.axvline(x_asymptote, linestyle="--")
# plt.text(x_asymptote * 1.01, y_top * 0.85,
#          r"$x=\frac{512}{3}$ Asymptote",
#          rotation=90, va="top")
# plt.text(x_asymptote + (x_max_plot - x_asymptote) * 0.25,
#          y_top * 0.15,
#          "No feasible region for inequality\nwhen x ≥ 512/3 (with y ≥ 0)",
#          ha="center")

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

plt.savefig("compare_absorb_or_not.png", dpi=300)
