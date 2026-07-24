# Two-figure version: one chart per routing method, with configurable EP and baseline.
# R in {1, 8, 64}, T = 1..512. Shows "max EP-bin (over bins, then over repeats)".
#
# Chart rules: matplotlib only, one plot per chart (each figure has multiple lines but a single axes).
# No explicit colors or styles.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mosaic.utils.estimate_moe_routing_max import estimate_moe_routing_max

# ---------------------- CONFIG ----------------------
M = 256
n = 8
EP = 8                 # 👈可改：例如 4, 8, 16, 32, 64, 128, 256
R_list = [1, 4, 32]
T_max = 128
rng_seed_base = 43

experts_per_bin = M // EP
expert_to_bin = (np.arange(M) // experts_per_bin).astype(np.int32)

groups = 8
experts_per_group = M // groups  # 32

# ---------------------- HELPERS ----------------------
def qwen_counts(conf_2d):
    """Qwen MoE routing: global top-8 per row -> per-EP-bin counts."""
    idx_top8 = np.argpartition(conf_2d, -n, axis=1)[:, -n:]  # (N,8)
    bins_top8 = expert_to_bin[idx_top8]
    N = conf_2d.shape[0]
    counts = np.zeros((N, EP), dtype=np.int32)
    row_idx = np.arange(N)[:, None]
    for k in range(n):
        np.add.at(counts, (row_idx[:, 0], bins_top8[:, k]), 1)
    return counts

def deepseek_counts(conf_2d):
    """
    DeepSeek routing for generic EP:
    - Split into 8 groups of 32.
    - For each group, take top-2 *indices and values*.
    - Compute group scores as sum of those values.
    - Select top-4 groups by score.
    - Selected experts = the top-2 indices of those 4 groups (total 8 experts).
    - Return per-EP-bin counts per row.
    """
    N = conf_2d.shape[0]
    g = conf_2d.reshape(N, groups, experts_per_group)          # (N,8,32)

    # top-2 local indices and values within each group
    # Using partition then gather to avoid full sort
    part_idx = np.argpartition(g, -2, axis=2)[:, :, -2:]       # (N,8,2) local indices
    # gather values
    row_ids = np.arange(N)[:, None, None]
    grp_ids = np.arange(groups)[None, :, None]
    top2_vals = g[row_ids, grp_ids, part_idx]                  # (N,8,2)
    group_scores = top2_vals.sum(axis=2)                       # (N,8)

    # top-4 groups by score
    top4_groups = np.argpartition(group_scores, -4, axis=1)[:, -4:]  # (N,4)

    # Convert local indices to global expert ids for the chosen groups
    # Prepare container for 8 picks (two per chosen group)
    chosen_bins = np.empty((N, n), dtype=np.int32)

    # We'll fill two columns at a time
    offset_per_group = (np.arange(groups) * experts_per_group).astype(np.int32)

    for col_pair, which_of_two in enumerate([0,1]):  # pick the 1st and 2nd top2 within each chosen group
        # get the local index (N,8 groups->we will subselect 4)
        local_idx = part_idx[:, :, which_of_two]                  # (N,8)
        # global expert id per (row, group)
        global_id_all_groups = offset_per_group[None, :] + local_idx  # (N,8)
        # take only the 4 chosen groups
        # build gather: for each row, top4_groups gives 4 group indices
        rows = np.arange(N)[:, None]
        global_id_top4 = global_id_all_groups[rows, top4_groups]      # (N,4)
        # map to bins
        chosen_bins[:, 4*0 + which_of_two : 4*0 + which_of_two + 1] = 0  # init; we'll fill below

        # place them in the appropriate positions in chosen_bins
        # we'll fill columns [2*0+which_of_two, 2*1+which_of_two, 2*2+which_of_two, 2*3+which_of_two]
        target_cols = [which_of_two, 2+which_of_two, 4+which_of_two, 6+which_of_two]
        for j, col in enumerate(target_cols):
            chosen_bins[:, col] = expert_to_bin[global_id_top4[:, j]]

    # Now accumulate to counts
    counts = np.zeros((N, EP), dtype=np.int32)
    row_idx_flat = np.arange(N)
    for k in range(n):
        np.add.at(counts, (row_idx_flat, chosen_bins[:, k]), 1)
    return counts

def run_extreme_curve(method:str, R:int, seed:int):
    rng = np.random.RandomState(seed)
    conf = rng.rand(R, T_max, M).astype(np.float32)  # (R, T, 256)
    batch = conf.reshape(-1, M)                      # (R*T, 256)
    if method == "Qwen":
        counts_flat = qwen_counts(batch)
    else:
        counts_flat = deepseek_counts(batch)
    counts = counts_flat.reshape(R, T_max, EP)
    cum = counts.cumsum(axis=1)                      # cumulative over T
    per_repeat_max = cum.max(axis=2)                 # max over EP bins
    extreme_curve = per_repeat_max.max(axis=0)       # max across repeats
    return extreme_curve

# ---------------------- RUN ----------------------
curves_qwen = {}
curves_deepseek = {}

for R in R_list:
    curves_qwen[R] = run_extreme_curve("Qwen", R, seed=rng_seed_base + 101 + R)
    curves_deepseek[R] = run_extreme_curve("DeepSeek", R, seed=rng_seed_base + 202 + R)

x = np.arange(1, T_max+1)
baseline = x * 8 / EP


# ---------------------- PLOTS ----------------------
# Single figure with two side-by-side subplots (left: Qwen, right: DeepSeek)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
ax_qwen, ax_deep = axes

# Left subplot: Qwen MoE routing
for R in R_list:
    line_empirical, = ax_qwen.plot(x, curves_qwen[R], label=f"Qwen MoE routing, R={R}")
    color = line_empirical.get_color()
    y_theory_qwen = np.array([estimate_moe_routing_max(M, n, EP, int(t), R, method="qwen") for t in x], dtype=np.float32)
    ax_qwen.plot(x, y_theory_qwen, "--", color=color, label=f"Qwen MoE Theoretical Expectation for R={R}")
ax_qwen.plot(x, baseline, "--", label=f"Expected mean per EP bin = 8T/{EP}")
ax_qwen.set_title(f"Max EP-bin vs T — Qwen MoE routing (EP={EP})")
ax_qwen.set_xlabel("T (tokens)")
ax_qwen.set_ylabel("Max EP-bin activations")
ax_qwen.legend()

# Right subplot: DeepSeek routing
for R in R_list:
    line_empirical, = ax_deep.plot(x, curves_deepseek[R], label=f"DeepSeek MoE routing, R={R}")
    color = line_empirical.get_color()
    y_theory_deep = np.array([estimate_moe_routing_max(M, n, EP, int(t), R, method="deepseek") for t in x], dtype=np.float32)
    ax_deep.plot(x, y_theory_deep, "--", color=color, label=f"DeepSeek MoE Theoretical Expectation for R={R}")
ax_deep.plot(x, baseline, "--", label=f"Expected mean per EP bin = 8T/{EP}")
ax_deep.set_title(f"Max EP-bin vs T — DeepSeek routing (EP={EP})")
ax_deep.set_xlabel("T (tokens)")
ax_deep.legend()

fig.tight_layout()
plt.savefig(f'ep_monte_calo_method_Qwen_DeepSeek_R_{R_list}.png', dpi=300)
plt.show()