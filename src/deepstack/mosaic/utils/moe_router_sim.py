import json
import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def simulate_token_routing(
    token_embedding: torch.Tensor,
    router_group_net: nn.Module,
    router_expert_net: nn.Module,
    config: dict
) -> dict:
    """Simulate node-limited routing for a single token, selecting groups by
    the sum of their top-k expert scores.

    Strategy:
      - Compute scores for all experts.
      - Sum each group's top k_per_group expert scores and select the
        topk_group groups by these sums.
      - Select k_per_group experts from each selected group, yielding
        num_experts_per_tok experts in total.
    """
    n_routed_experts = config["n_routed_experts"]
    n_group = config["n_group"]
    topk_group = config["topk_group"]
    num_experts_per_tok = config["num_experts_per_tok"]

    assert n_routed_experts % n_group == 0, "n_routed_experts 必须能被 n_group 整除"
    experts_per_group = n_routed_experts // n_group

    assert num_experts_per_tok % topk_group == 0, \
        "num_experts_per_tok 必须能被 topk_group 整除"
    k_per_group = num_experts_per_tok // topk_group

    all_expert_scores = router_expert_net(token_embedding).squeeze(0)  # [N_experts]

    group_scores = []
    group_top_indices = []
    for g in range(n_group):
        start = g * experts_per_group
        end = start + experts_per_group
        group_slice_scores = all_expert_scores[start:end]  # [experts_per_group]

        top_vals, top_idx_local = torch.topk(group_slice_scores, k_per_group)
        group_scores.append(top_vals.sum().item())

        top_idx_global = (top_idx_local + start).tolist()
        group_top_indices.append(top_idx_global)

    group_scores_tensor = torch.tensor(group_scores, device=token_embedding.device)
    _, top_group_indices = torch.topk(group_scores_tensor, topk_group)
    selected_groups_list = sorted(top_group_indices.tolist())

    selected_experts = []
    for g in selected_groups_list:
        selected_experts.extend(group_top_indices[g])

    assert len(selected_experts) == num_experts_per_tok, \
        f"选出的专家数量 {len(selected_experts)} 与 num_experts_per_tok 不一致"

    selected_experts_list = sorted(selected_experts)

    return {
        "selected_experts": selected_experts_list,
        "selected_groups": selected_groups_list,
        "k_per_group": k_per_group
    }

def run_simulation(config: dict):
    """Run the complete routing simulation using the configuration."""
    print("开始新的模拟计算...")
    torch.manual_seed(42)

    embedding_dim = 2048
    router_group_net = nn.Linear(embedding_dim, config["n_group"])
    router_expert_net = nn.Linear(embedding_dim, config["n_routed_experts"])

    results = {}

    num_tokens = config["iter"]
    for i in tqdm(range(num_tokens), desc="模拟Token路由"):
        token_embedding = torch.randn(1, embedding_dim)
        routing_info = simulate_token_routing(
            token_embedding, router_group_net, router_expert_net, config
        )
        results[f"token_{i}"] = routing_info

    print("模拟计算完成。")
    return results

def generate_filename(config: dict) -> str:
    """Generate a deterministic filename from the configuration."""
    return (
        f"routing_iter{config['iter']}"
        f"_N{config['n_routed_experts']}"
        f"_G{config['n_group']}"
        f"_TG{config['topk_group']}"
        f"_E{config['num_experts_per_tok']}.json"
    )

def export_results(filepath: str, config: dict, data: dict):
    """Export the configuration and results to a JSON file."""
    output_data = {
        "configuration": config,
        "routing_results": data
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"结果已成功导出到: {filepath}")

def import_results(filepath: str) -> dict:
    """Import results from a JSON file."""
    print(f"从现有文件加载结果: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_selected_experts_lists(routing_results: dict, n: Optional[int] = None) -> list:
    """Extract expert ID lists from routing_results in token order.

    Example return value: [[1, 5, 6, 78, ...], [13, 53, ...], ...]
    """
    def token_order(key: str) -> int:
        try:
            return int(key.split('_')[-1])
        except Exception:
            return 10**9

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n 必须为正整数")

    expert_id_lists = []
    for key in sorted(routing_results.keys(), key=token_order):
        entry = routing_results[key]
        if "selected_experts" not in entry:
            raise KeyError(f"routing_results['{key}'] 缺少 'selected_experts'")
        expert_id_lists.append(list(entry["selected_experts"]))
        if n is not None and len(expert_id_lists) >= n:
            return expert_id_lists
    return expert_id_lists

def import_results_as_expert_id_lists(filepath: str, n: Optional[int] = None) -> list:
    """Read a JSON file exported by this script and return a two-dimensional list,
    where each sublist contains the expert IDs to which a token was routed.

    Args:
        n: Optional number of initial routed lists to return. If omitted, return all.
            Raise ValueError if fewer than n lists are available.

    Examples:
        n=1 -> [[6, 19, 98, 109, 142, 156, 204, 216]]
        n=2 -> [[...], [...]]
    """
    if n is None:
        loaded = import_results(filepath)
        routing_results = loaded.get("routing_results", {})
        return extract_selected_experts_lists(routing_results)

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n 必须为正整数")

    try:
        import ijson

        expert_id_lists = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for key, entry in ijson.kvitems(f, 'routing_results'):
                if "selected_experts" not in entry:
                    raise KeyError(f"routing_results['{key}'] 缺少 'selected_experts'")
                expert_id_lists.append(list(entry["selected_experts"]))
                if len(expert_id_lists) >= n:
                    return expert_id_lists

        raise ValueError(
            f"请求的前 {n} 个 routed list，但实际只有 {len(expert_id_lists)} 个"
        )
    except ImportError:
        loaded = import_results(filepath)
        routing_results = loaded.get("routing_results", {})
        expert_id_lists = extract_selected_experts_lists(routing_results, n=n)
        if len(expert_id_lists) < n:
            raise ValueError(
                f"请求的前 {n} 个 routed list，但实际只有 {len(expert_id_lists)} 个"
            )
        return expert_id_lists
    except Exception:
        loaded = import_results(filepath)
        routing_results = loaded.get("routing_results", {})
        expert_id_lists = extract_selected_experts_lists(routing_results, n=n)
        if len(expert_id_lists) < n:
            raise ValueError(
                f"请求的前 {n} 个 routed list，但实际只有 {len(expert_id_lists)} 个"
            )
        return expert_id_lists



def load_npz_routing_flatten_last_dim(npz_path: str, as_list: bool = False, expected_last_dim: int = 8):
    """Read a .npz file containing 'prefill' and 'decode' and flatten to
    shape [N, expected_last_dim].

    Expected layouts:
      - prefill: [layer, iter, num_activated_tokens]
      - decode: [iter, layer, batch, num_activated_tokens]

    Flattening rules:
      - Flatten all dimensions except the last in C order, preserving the final
        dimension of length expected_last_dim.
      - If both arrays exist, concatenate prefill before decode after flattening.

    Args:
        npz_path: Path to the .npz file.
        as_list: If True, return List[List[...]]; otherwise return numpy.ndarray.
        expected_last_dim: Expected size of the final dimension; defaults to 8.

    Returns:
        numpy.ndarray of shape [N, expected_last_dim] by default,
        or List[List[...]] if as_list=True.
    """

    def _reshape_to_rows(arr: np.ndarray, name: str) -> np.ndarray:
        if arr is None:
            return np.empty((0, expected_last_dim), dtype=np.float32)
        if arr.dtype == object:
            try:
                numeric = np.asarray(arr, dtype=np.int64)
                if numeric.ndim >= 1 and numeric.shape[-1] == expected_last_dim:
                    return numeric.reshape(-1, expected_last_dim)
                flat_num = numeric.reshape(-1)
                if flat_num.size % expected_last_dim != 0:
                    raise ValueError(
                        f"{name} 展平后长度 {flat_num.size} 不能被 {expected_last_dim} 整除，原形状 {numeric.shape}"
                    )
                return flat_num.reshape(-1, expected_last_dim)
            except Exception:
                flat_obj = np.asarray(arr, dtype=object).ravel()
                collected = []
                for idx, item in enumerate(flat_obj):
                    try:
                        collected.append(int(item))
                    except Exception:
                        sub = np.asarray(item)
                        if sub.ndim >= 1 and sub.shape[-1] == expected_last_dim:
                            collected.extend(sub.reshape(-1))
                        else:
                            raise ValueError(
                                f"{name} 中第 {idx} 项既不是标量也不是最后一维为 {expected_last_dim} 的数组，shape={sub.shape}"
                            )
                if len(collected) == 0:
                    return np.empty((0, expected_last_dim), dtype=np.float32)
                if len(collected) % expected_last_dim != 0:
                    raise ValueError(
                        f"{name} 收集到的元素数 {len(collected)} 不能被 {expected_last_dim} 整除"
                    )
                return np.asarray(collected, dtype=np.int64).reshape(-1, expected_last_dim)
        if arr.ndim >= 1 and arr.shape[-1] == expected_last_dim:
            return arr.reshape(-1, expected_last_dim)
        flat = np.asarray(arr).reshape(-1)
        if flat.size % expected_last_dim != 0:
            raise ValueError(
                f"{name} 展平后长度 {flat.size} 不能被 {expected_last_dim} 整除，原形状 {arr.shape}"
            )
        return flat.reshape(-1, expected_last_dim)

    with np.load(npz_path, allow_pickle=True) as zf:
        prefill = zf.get("prefill", None)
        decode = zf.get("decode", None)

    if prefill is None and decode is None:
        raise KeyError("npz 中未找到 'prefill' 或 'decode' 任一键")

    prefill_rows = _reshape_to_rows(prefill, "prefill") if prefill is not None else np.empty((0, expected_last_dim), dtype=np.float32)
    decode_rows = _reshape_to_rows(decode, "decode") if decode is not None else np.empty((0, expected_last_dim), dtype=prefill_rows.dtype)

    # rows = np.concatenate([prefill_rows, decode_rows], axis=0) if (len(prefill_rows) or len(decode_rows)) else prefill_rows

    # if as_list:
    #     return rows.tolist()
    # return rows
    if as_list:
        return prefill_rows.tolist(), decode_rows.tolist()
    return prefill_rows, decode_rows


def load_npz_routing_keep_shape(npz_path: str, as_list: bool = False, expected_last_dim: int = 8):
    """Read 'prefill' and 'decode' from a .npz file, preserving their original shapes:
      - prefill is expected to have shape [layer, iter, expected_last_dim].
      - decode is expected to have shape [iter, layer, batch, expected_last_dim].

    If either array has dtype=object, attempt to convert the entire array to a
    numeric array. Assert that both final dimensions equal expected_last_dim
    (default 8).

    Returns:
        A tuple of numpy.ndarray objects (prefill, decode) by default.
        If as_list=True, return (prefill.tolist(), decode.tolist()).
    """

    # 

    def _to_numeric_keep_shape(arr: np.ndarray, name: str) -> np.ndarray:
        if arr is None:
            return None
        if arr.dtype == object:
            arr_num = np.asarray(arr, dtype=np.int64)
        else:
            arr_num = np.asarray(arr)

        if arr_num.ndim < 1 or arr_num.shape[-1] != expected_last_dim:
            raise AssertionError(
                f"{name} 的最后一维应为 {expected_last_dim}，实际为 {arr_num.shape[-1]}，形状为 {arr_num.shape}"
            )
        return arr_num

    with np.load(npz_path, allow_pickle=True) as zf:
        prefill = zf.get("prefill", None)
        decode = zf.get("decode", None)

    if prefill is None and decode is None:
        raise KeyError("npz 中未找到 'prefill' 或 'decode' 任一键")

    prefill_num = _to_numeric_keep_shape(prefill, "prefill") if prefill is not None else None
    decode_num = _to_numeric_keep_shape(decode, "decode") if decode is not None else None

    if as_list:
        return (
            prefill_num.tolist() if prefill_num is not None else None,
            decode_num.tolist() if decode_num is not None else None,
        )
    return prefill_num, decode_num

def example_gen_routing_list():
    simulation_config = {
        "iter": 128,
        "n_routed_experts": 256,
        "n_group": 8,
        "topk_group": 4,
        "num_experts_per_tok": 8,
        "output_dir": "../data/routing_outputs"
    }
    # =================================================

    filename = generate_filename(simulation_config)
    filepath = os.path.join(simulation_config["output_dir"], filename)

    if os.path.exists(filepath):
        print(f"发现已存在的配置文件: {filename}")
        loaded_data = import_results(filepath)

        print("\n加载的数据预览：")
        print("配置:", loaded_data['configuration'])
        token_0_result = loaded_data['routing_results']['token_0']
        print("Token 0 的路由结果:")
        print(f"  - 选择的专家: {token_0_result['selected_experts']}")
        print(f"  - 来源的组:   {token_0_result['selected_groups']}")
        if 'k_per_group' in token_0_result:
            print(f"  - 每组专家数: {token_0_result['k_per_group']}")

        expert_id_lists = import_results_as_expert_id_lists(filepath)
        print("\n二维列表示例（前2个 token）：")
        for idx in range(min(2, len(expert_id_lists))):
            print(f"  token_{idx}: {expert_id_lists[idx]}")
    else:
        print(f"未找到匹配的配置文件，将创建新的模拟: {filename}")
        routing_data = run_simulation(simulation_config)

        export_results(filepath, simulation_config, routing_data)

        print("\n生成的数据预览：")
        token_0_result = routing_data['token_0']
        print("Token 0 的路由结果:")
        print(f"  - 选择的专家: {token_0_result['selected_experts']}")
        print(f"  - 来源的组:   {token_0_result['selected_groups']}")
        print(f"  - 每组专家数: {token_0_result['k_per_group']}")

        expert_id_lists = extract_selected_experts_lists(routing_data)
        print("\n二维列表示例（前2个 token）：")
        for idx in range(min(2, len(expert_id_lists))):
            print(f"  token_{idx}: {expert_id_lists[idx]}")

def extract_prefill_and_decode_from_npz():
    npz_path = "../data/routing/deepseek_v3.npz"
    prefill_list, decode_list = load_npz_routing_flatten_last_dim(npz_path,as_list=False)
    print(prefill_list.shape)
    print(decode_list.shape)
    print(prefill_list[0])
    print(decode_list[0])
    return prefill_list, decode_list

def extract_prefill_and_decode_from_npz_as_list():
    # npz_path = "../data/routing/deepseek_v3.npz"
    npz_path = "../data/qwen_e128_a128/moe_activations_k128.npz"


    

    prefill_list, decode_list = load_npz_routing_keep_shape(npz_path,as_list=False, expected_last_dim=128)
    print(prefill_list.shape)
    print(decode_list.shape)
    # print(prefill_list[0])
    print(decode_list[0,0,0])
    # min_value = decode_list.min()
    # max_value = decode_list.max()
    return prefill_list, decode_list

if __name__ == "__main__":

    
    # example_gen_routing_list()
    # prefill_list, decode_list = extract_prefill_and_decode_from_npz()
    prefill_list, decode_list = extract_prefill_and_decode_from_npz_as_list()


