# illustration https://zhuanlan.zhihu.com/p/1941992212989732340
import numpy as np
import matplotlib.pyplot as plt

def y_boundary(x):
    # y = 3x^2 / (512 - 3x) 仅在 x < 512/3 且 y>=0 有意义
    return (3.0 * x**2) / (512.0 - 3.0 * x)

# ---- 主参数 ----
x_max_plot = 512
x_asymptote = 512.0 / 3.0  # ≈170.6667

# 整体 x 轴用于显示（0~8192），但曲线与填充只在 x < 渐近线处绘制
x = np.linspace(0.0, x_max_plot, 5000)

# 仅保留 x < 渐近线 的有效部分用于曲线/填充
mask_valid = x < x_asymptote
x_valid = x[mask_valid]

# 边界曲线（无效区域用 NaN 避免 fill_between 误填）
y_b_valid = y_boundary(x_valid)

# 选择一个合理的 y 上限用于可视化（避免接近渐近线时无限大）
# 这里取在渐近线 99% 处的 y 值作为参考上限
x_ref = x_asymptote * 0.97
y_ref = y_boundary(x_ref)
y_top = float(y_ref) * 1.05 if np.isfinite(y_ref) and y_ref > 0 else 1e6

# ---- 绘图 ----
# 统一字体大小
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

plt.figure(figsize=(6, 4))
ax = plt.gca()

# 1) 边界曲线（仅绘制有效区域）
# plt.plot(x_valid, y_b_valid, label=r"Boundary: $512y = 3x^2 + 3xy$")
plt.plot(x_valid, y_b_valid, label=r"Boundary")

# 2) 仅在有效区域内填充满足区域： y > y_boundary(x)
y_fill_top = np.full_like(x_valid, y_top)
plt.fill_between(x_valid, y_b_valid, y_fill_top,
                 where=(y_fill_top > y_b_valid),
                 alpha=0.2, label="MLA absorbs better")

# 3) 在无可行区域的 x≥渐近线 位置做注释
# plt.axvline(x_asymptote, linestyle="--")
# plt.text(x_asymptote * 1.01, y_top * 0.85,
#          r"$x=\frac{512}{3}$ Asymptote",
#          rotation=90, va="top")
# plt.text(x_asymptote + (x_max_plot - x_asymptote) * 0.25,
#          y_top * 0.15,
#          "No feasible region for inequality\nwhen x ≥ 512/3 (with y ≥ 0)",
#          ha="center")

# 4) 其他图形元素
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

# 保存并展示
plt.savefig("compare_absorb_or_not.png", dpi=300)
