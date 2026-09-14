import numpy as np
import matplotlib.pyplot as plt
from mosaic.utils.moe_router_sim import load_npz_routing_keep_shape
import logging
log = logging.getLogger(__name__) 


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

def count_expert_frequency_flatten(flat_numpy: np.ndarray, total_experts: int, group_tokens: int) -> np.ndarray:
    """Count expert activations in an array of shape [tokens, num_activated_experts].
    Group tokens into windows of group_tokens and count frequencies separately
    for each group. Return a matrix of shape [num_groups, total_experts], where
    num_groups = ceil(tokens / group_tokens).
    Include the final group even if it contains fewer than group_tokens tokens.
    """
    assert flat_numpy.ndim == 2, "输入应为二维 [tokens, num_activated_experts]"
    assert group_tokens > 0, "group_tokens 必须为正整数"
    tokens, k = flat_numpy.shape
    # print(f"tokens: {tokens}, k: {k}, group_tokens: {group_tokens}")
    num_groups = (tokens + group_tokens - 1) // group_tokens
    # print(f"num_groups: {num_groups}")
    freq_matrix = np.zeros((num_groups, total_experts), dtype=int)
    for g in range(num_groups):
        start = g * group_tokens
        end = min(start + group_tokens, tokens)
        for t in range(start, end):
            for expert in flat_numpy[t]:
                if 0 <= expert < total_experts:
                    freq_matrix[g, expert] += 1
    # print(f"freq_matrix.shape: {freq_matrix.shape}")
    # print(f"freq_matrix: {freq_matrix}")
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


def count_expert_frequency_flatten_wrapper(flatten_numpy: np.ndarray, group_tokens: int, total_experts: int, EP: int) -> np.ndarray:
    """Count expert activations by token group in [tokens, num_activated_experts]
    and aggregate by EP device. Return a matrix of shape [num_groups, EP], where
    num_groups = ceil(tokens/group_tokens).

    expert_row contains each expert's activation count within the expert partition
    for (group_idx, ep_idx), indexed from 0 to total_experts/ep-1.
    It has shape [total_experts/EP], and its sum equals np.max(freq_groups).
    """

    assert flatten_numpy.ndim == 2, "flatten_numpy 应为二维 [tokens, num_activated_experts]"
    num_tokens, num_activated_experts = flatten_numpy.shape

    # First count by expert: [1, total_experts] per group, [num_groups, total_experts] overall
    # num_groups = ceil(tokens/group_tokens)
    freq = count_expert_frequency_flatten(flatten_numpy, total_experts, group_tokens)
    # print(freq.shape)
    

    # Then aggregate by EP device: [total_experts] -> [EP] per group
    group_ids = build_group_index(total_experts, EP)
    freq_groups  = aggregate_freq_by_groups(freq, group_ids, EP)  # [num_groups, EP]
    # print(freq_groups.shape)
    # log.info("freq.shape: %s", freq.shape)
    # log.info("freq_groups.shape: %s", freq_groups.shape)
    # log.info("freq_groups: %s", freq_groups)

    expected_freq = (group_tokens * num_activated_experts) / total_experts 
    expected_group_freq = (group_tokens * num_activated_experts) / EP

    repeat_times =freq_groups.shape[0]
    
    log.info("tokens per ep parallel group: %s", group_tokens)
    log.info("expected each expert activated times: %s, expected each device activated times: %s", expected_freq, expected_group_freq)
    log.info("repeat_times(dp * sp): %s", repeat_times)


    # Find the maximum value in this freq_groups and return its index
    max_activated_experts_times_per_device = np.max(freq_groups)
    log.info("max activated tokens per device: %s", max_activated_experts_times_per_device)

    expert_row = extract_expert_row_for_max(flatten_numpy, group_tokens, total_experts, EP)

    log.info("expert_row: %s", expert_row)
    
    # expert_row_sum = expert_row.sum()
    # log.info("expert_row_sum: %s", expert_row_sum)


    return expert_row


def extract_expert_row_for_max(
    flatten_numpy: np.ndarray,
    group_tokens: int,
    total_experts: int,
    EP: int,
):
    """Return the per-expert detail row at the global maximum of freq_groups:
      - group_idx: Group index after grouping by group_tokens.
      - ep_idx: Device index (0..EP-1).
      - expert_row: Activation count for each expert in the partition corresponding
        to (group_idx, ep_idx). Shape [total_experts/EP]; its sum equals np.max(freq_groups).
      - freq_groups: Shape [num_groups, EP].

    Preserves the existing wrapper's return format to avoid breaking callers.
    Call this function directly when expert details are needed.
    """
    assert flatten_numpy.ndim == 2, "flatten_numpy 应为二维 [tokens, num_activated_experts]"
    assert group_tokens > 0, "group_tokens 必须为正整数"
    assert total_experts % EP == 0, "total_experts 应能被 EP 整除"

    # First obtain expert frequencies per group: [num_groups, total_experts]
    freq = count_expert_frequency_flatten(flatten_numpy, total_experts, group_tokens)
    # Aggregate into the device dimension using group_ids
    group_ids = build_group_index(total_experts, EP)
    freq_groups = aggregate_freq_by_groups(freq, group_ids, EP)  # [num_groups, EP]

    # Find the global maximum (group_idx, ep_idx)
    flat_idx = np.argmax(freq_groups)
    group_idx, ep_idx = np.unravel_index(flat_idx, freq_groups.shape)

    # Select the expert-partition columns for this device and return that group's per-expert details
    cols = (group_ids == ep_idx)
    expert_row = freq[group_idx, cols]

    # Verify that the sum equals the aggregate value at this position
    if int(expert_row.sum()) != int(freq_groups[group_idx, ep_idx]):
        log.warning("expert_row 求和与聚合值不一致: %s != %s", expert_row.sum(), freq_groups[group_idx, ep_idx])

    # log.info("group_idx: %s, ep_idx: %s, expert_row: %s, freq_groups: %s", group_idx, ep_idx, expert_row, freq_groups)

    # return group_idx, ep_idx, expert_row, freq_groups
    return expert_row

# ===================== Main workflow =====================

if __name__ == "__main__":
    # 1) Read data.
    npz_path = "../data/aime_ds_r1/moe_activations_batch0.npz"
    prefill_numpy, decode_numpy = load_npz_routing_keep_shape(npz_path, as_list=False)
    logging.basicConfig(
        level=logging.INFO,                              # Global logging level.
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # prefill: [num_layers, num_tokens, k]
    # decode : [num_iters, num_layers, batch, k]

    # ===== Configurable parameters =====
    model_name = "DeepSeek-R1"   # <- Supply the model name here.
    total_experts = 256            # Total number of experts on the X axis.
    EP = 16                         # Number of experts activated each time (also the number of device groups).

    group_tokens = 1024

    routing_array = decode_numpy.reshape(-1, decode_numpy.shape[-1])

    # Take the first 512
    routing_array = routing_array[:1024]
    # print(routing_array.shape)
    # print(routing_array)

    # group_tokens=tokens_per_ep_group, total_experts=num_routed_experts, EP=parallel.ep)
    freq_groups = count_expert_frequency_flatten_wrapper(routing_array, group_tokens, total_experts, EP)

    log.info("freq_groups: %s", freq_groups)
    log.info("freq_groups.shape: %s", freq_groups.shape)


