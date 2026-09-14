import numpy as np
import matplotlib.pyplot as plt
from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape

# ===================== Statistics functions =====================

def count_expert_frequency_prefill(prefill_numpy, total_experts):
    num_layers = prefill_numpy.shape[0]
    freq_matrix = np.zeros((num_layers, total_experts), dtype=int)
    for layer in range(num_layers):
        for token_experts in prefill_numpy[layer]:
            for expert in token_experts:
                if expert < total_experts:
                    freq_matrix[layer, expert] += 1
    return freq_matrix

def count_expert_frequency_decode(decode_numpy, total_experts):
    num_iters, num_layers, batch, k = decode_numpy.shape
    freq_matrix = np.zeros((num_iters * num_layers, total_experts), dtype=int)
    idx = 0
    for it in range(num_iters):
        for layer in range(num_layers):
            for b in range(batch):
                for expert in decode_numpy[it, layer, b]:
                    if expert < total_experts:
                        freq_matrix[idx, expert] += 1
            idx += 1
    return freq_matrix

# ===================== Truncation functions =====================

def truncate_decode(decode_numpy, new_num_iters):
    assert decode_numpy.ndim == 4, "decode_numpy 必须是 4 维的"
    original_iters = decode_numpy.shape[0]
    if new_num_iters > original_iters or new_num_iters <= 0:
        raise ValueError(f"new_num_iters 必须在 1 到 {original_iters} 之间")
    return decode_numpy[:new_num_iters]

def truncate_prefill(prefill_numpy, new_num_tokens):
    assert prefill_numpy.ndim == 3, "prefill_numpy 必须是 3 维的"
    original_tokens = prefill_numpy.shape[1]
    if new_num_tokens > original_tokens or new_num_tokens <= 0:
        raise ValueError(f"new_num_tokens 必须在 1 到 {original_tokens} 之间")
    return prefill_numpy[:, :new_num_tokens, :]

# ===================== Grouped aggregation (EP devices) =====================

def build_group_index(total_experts: int, EP: int) -> np.ndarray:
    expert_ids = np.arange(total_experts)
    group_ids = (expert_ids * EP) // total_experts
    return np.clip(group_ids, 0, EP - 1)

def aggregate_freq_by_groups(freq_matrix: np.ndarray, group_ids: np.ndarray, EP: int) -> np.ndarray:
    N, total_experts = freq_matrix.shape
    assert group_ids.shape[0] == total_experts
    out = np.zeros((N, EP), dtype=freq_matrix.dtype)
    for g in range(EP):
        cols = (group_ids == g)
        if np.any(cols):
            out[:, g] = freq_matrix[:, cols].sum(axis=1)
    return out

# ===================== Visualization =====================

def _safe_title_to_filename(title: str) -> str:
    bad = [' ', '(', ')', ':', '/', '\\', '|', '=']
    for ch in bad:
        title = title.replace(ch, '_')
    return title

def plot_heatmap(data, title, xlabel="Expert ID", ylabel="Layer / Step Index", expected_value=None):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data, aspect='auto', cmap='viridis')
    cb = plt.colorbar(im, label='Frequency')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if expected_value is not None:
        vmin, vmax = im.get_clim()
        if vmin <= expected_value <= vmax:
            cb.ax.hlines(expected_value, 0, 1, colors='red', linestyles='--', linewidth=2)
            cb.ax.text(1.05, expected_value, f"Expected={expected_value:.2f}",
                       transform=cb.ax.get_yaxis_transform(), color='red', va='center')

    plt.tight_layout()
    plt.savefig(f"{_safe_title_to_filename(title)}.png", dpi=300)
    plt.show()

# ===================== Main workflow =====================

if __name__ == "__main__":
    # 1) Read data.
    npz_path = "./moe_activations_batch0.npz"
    prefill_numpy, decode_numpy = load_npz_routing_keep_shape(npz_path, as_list=False)
    # prefill: [num_layers, num_tokens, k]
    # decode : [num_iters, num_layers, batch, k]

    # ===== Configurable parameters =====
    model_name = "DeepSeek-R1"   # <- Supply the model name here.
    total_experts = 256            # Total number of experts on the X axis.
    EP = 8                         # Number of experts activated each time (also the number of device groups).

    # ===== Optional: truncation (enable as needed) =====
    decode_numpy = truncate_decode(decode_numpy, new_num_iters=2)
    # prefill_numpy = truncate_prefill(prefill_numpy, new_num_tokens=1000)

    # 2) Basic shape information.
    num_layers_prefill, num_tokens, k_prefill = prefill_numpy.shape
    num_iters, num_layers_decode, batch, k_decode = decode_numpy.shape

    # 3) Statistics (per expert).
    prefill_freq = count_expert_frequency_prefill(prefill_numpy, total_experts)   # [num_layers, total_experts]
    decode_freq  = count_expert_frequency_decode(decode_numpy, total_experts)     # [num_iters*num_layers, total_experts]

    # 3.1 Expected values (per expert).
    expected_prefill_expert = (num_tokens * k_prefill) / total_experts
    expected_decode_expert  = (batch * k_decode) / total_experts

    # 3.2 Plot (per expert; include model and batch in the title).
    plot_heatmap(
        prefill_freq,
        title=f"Prefill Expert Activation Frequencies | Model={model_name} | Batch={batch} | Total Experts={total_experts}",
        ylabel="Layer Index",
        xlabel="Expert ID",
        expected_value=expected_prefill_expert
    )

    plot_heatmap(
        decode_freq,
        title=f"Decode Expert Activation Frequencies | Model={model_name} | Batch={batch} | Total Experts={total_experts}",
        ylabel="Iter×Layer Index",
        xlabel="Expert ID",
        expected_value=expected_decode_expert
    )

    # 4) Group into EP devices.
    group_ids = build_group_index(total_experts, EP)
    prefill_freq_groups = aggregate_freq_by_groups(prefill_freq, group_ids, EP)   # [num_layers, EP]
    decode_freq_groups  = aggregate_freq_by_groups(decode_freq,  group_ids, EP)   # [num_iters*num_layers, EP]

    # 4.1 Expected values (EP groups).
    expected_prefill_group = (num_tokens * k_prefill) / EP
    expected_decode_group  = (batch * k_decode) / EP

    # 4.2 Plot (EP groups; include model and batch in the title).
    plot_heatmap(
        prefill_freq_groups,
        title=f"Prefill Activation per EP Device | Model={model_name} | Batch={batch} | EP={EP}",
        ylabel="Layer Index",
        xlabel=f"EP Device (0..{EP-1})",
        expected_value=expected_prefill_group
    )

    plot_heatmap(
        decode_freq_groups,
        title=f"Decode Activation per EP Device | Model={model_name} | Batch={batch} | EP={EP}",
        ylabel="Iter×Layer Index",
        xlabel=f"EP Device (0..{EP-1})",
        expected_value=expected_decode_group
    )
