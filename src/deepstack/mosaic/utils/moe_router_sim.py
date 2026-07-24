import json
import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm  # 用于显示进度条，可通过 pip install tqdm 安装

# --- 1. 核心路由模拟函数 ---
def simulate_token_routing(
    token_embedding: torch.Tensor,
    router_group_net: nn.Module,  # 不再用于打分，但为兼容保留参数
    router_expert_net: nn.Module,
    config: dict
) -> dict:
    """
    为单个Token模拟“节点受限路由”的决策过程（基于组内Top-k之和选择组）。

    策略：
      - 计算所有专家分数；
      - 每组取组内 top k_per_group 的专家分数并求和，按该和选出 topk_group 个组；
      - 从这些组里各取 k_per_group 个专家，最终得到 num_experts_per_tok 个专家。
    """
    n_routed_experts = config["n_routed_experts"]
    n_group = config["n_group"]
    topk_group = config["topk_group"]
    num_experts_per_tok = config["num_experts_per_tok"]

    # 每组专家数
    assert n_routed_experts % n_group == 0, "n_routed_experts 必须能被 n_group 整除"
    experts_per_group = n_routed_experts // n_group

    # 确定每组要选几个专家（要求能整除）
    assert num_experts_per_tok % topk_group == 0, \
        "num_experts_per_tok 必须能被 topk_group 整除"
    k_per_group = num_experts_per_tok // topk_group

    # 仅使用专家路由网络的分数
    all_expert_scores = router_expert_net(token_embedding).squeeze(0)  # [N_experts]

    # 计算每个组的聚合分数（组内 top k_per_group 之和）
    group_scores = []
    group_top_indices = []  # 记录每个组内被选中的专家索引（全局索引）
    for g in range(n_group):
        start = g * experts_per_group
        end = start + experts_per_group
        group_slice_scores = all_expert_scores[start:end]  # [experts_per_group]

        # 取组内 top k_per_group 专家（返回的是相对索引）
        top_vals, top_idx_local = torch.topk(group_slice_scores, k_per_group)
        group_scores.append(top_vals.sum().item())

        # 转成全局专家索引并保存
        top_idx_global = (top_idx_local + start).tolist()
        group_top_indices.append(top_idx_global)

    # 选出聚合分数最高的 topk_group 个组
    group_scores_tensor = torch.tensor(group_scores, device=token_embedding.device)
    _, top_group_indices = torch.topk(group_scores_tensor, topk_group)
    selected_groups_list = sorted(top_group_indices.tolist())

    # 从这些组中取出各自记录的 k_per_group 个专家索引
    selected_experts = []
    for g in selected_groups_list:
        selected_experts.extend(group_top_indices[g])

    # 安全检查：数量应当正好为 num_experts_per_tok
    assert len(selected_experts) == num_experts_per_tok, \
        f"选出的专家数量 {len(selected_experts)} 与 num_experts_per_tok 不一致"

    selected_experts_list = sorted(selected_experts)

    return {
        "selected_experts": selected_experts_list,
        "selected_groups": selected_groups_list,
        "k_per_group": k_per_group
    }

# --- 2. 主模拟流程 ---
def run_simulation(config: dict):
    """
    根据配置运行完整的路由模拟。
    """
    print("开始新的模拟计算...")
    torch.manual_seed(42)

    embedding_dim = 2048
    router_group_net = nn.Linear(embedding_dim, config["n_group"])            # 兼容保留
    router_expert_net = nn.Linear(embedding_dim, config["n_routed_experts"])  # 关键打分网络

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

# --- 3. 文件处理函数 ---
def generate_filename(config: dict) -> str:
    """根据配置生成确定性的文件名。"""
    return (
        f"routing_iter{config['iter']}"
        f"_N{config['n_routed_experts']}"
        f"_G{config['n_group']}"
        f"_TG{config['topk_group']}"
        f"_E{config['num_experts_per_tok']}.json"
    )

def export_results(filepath: str, config: dict, data: dict):
    """将配置和结果数据导出到JSON文件。"""
    output_data = {
        "configuration": config,
        "routing_results": data
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"结果已成功导出到: {filepath}")

def import_results(filepath: str) -> dict:
    """从JSON文件导入结果。"""
    print(f"从现有文件加载结果: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 将 routing_results 转换为二维列表，每个子列表是对应 token 的专家 ID 列表
def extract_selected_experts_lists(routing_results: dict, n: Optional[int] = None) -> list:
    """
    从 routing_results 字典中提取按 token 顺序排列的专家 ID 列表。

    返回示例：[[1, 5, 6, 78, ...], [13, 53, ...], ...]
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
    """
    读取由本脚本导出的 JSON 文件，并返回二维列表：
    每个子列表表示该 token 被路由到的专家 ID 列表。

    参数：
      - n: 可选，返回前 n 个 routed list；
           不传则返回全部；若实际数量少于 n 则抛出 ValueError。
    示例：
      - n=1 -> [[6, 19, 98, 109, 142, 156, 204, 216]]
      - n=2 -> [[...], [...]]
    """
    if n is None:
        loaded = import_results(filepath)
        routing_results = loaded.get("routing_results", {})
        return extract_selected_experts_lists(routing_results)

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n 必须为正整数")

    # 优先尝试流式解析，避免加载整个 JSON
    try:
        import ijson  # 可选依赖：pip install ijson

        expert_id_lists = []
        with open(filepath, 'r', encoding='utf-8') as f:
            # 逐个 key/value 迭代 routing_results 下的条目
            for key, entry in ijson.kvitems(f, 'routing_results'):
                if "selected_experts" not in entry:
                    raise KeyError(f"routing_results['{key}'] 缺少 'selected_experts'")
                expert_id_lists.append(list(entry["selected_experts"]))
                if len(expert_id_lists) >= n:
                    return expert_id_lists

        # 流式读取结束但不足 n，抛错
        raise ValueError(
            f"请求的前 {n} 个 routed list，但实际只有 {len(expert_id_lists)} 个"
        )
    except ImportError:
        # 回退：一次性读取（仍只构造前 n 个）
        loaded = import_results(filepath)
        routing_results = loaded.get("routing_results", {})
        expert_id_lists = extract_selected_experts_lists(routing_results, n=n)
        if len(expert_id_lists) < n:
            raise ValueError(
                f"请求的前 {n} 个 routed list，但实际只有 {len(expert_id_lists)} 个"
            )
        return expert_id_lists
    except Exception:
        # 任何流式异常都回退到安全路径
        loaded = import_results(filepath)
        routing_results = loaded.get("routing_results", {})
        expert_id_lists = extract_selected_experts_lists(routing_results, n=n)
        if len(expert_id_lists) < n:
            raise ValueError(
                f"请求的前 {n} 个 routed list，但实际只有 {len(expert_id_lists)} 个"
            )
        return expert_id_lists



# --- 3.1 从NPZ读取并展平成[:, expected_last_dim] ---
def load_npz_routing_flatten_last_dim(npz_path: str, as_list: bool = False, expected_last_dim: int = 8):
    """
    从包含 'prefill' 与 'decode' 的 .npz 文件读取并展平为形状 [N, expected_last_dim]。

    约定：
      - prefill: [layer, iter, num_activated_tokens]
      - decode:  [iter, layer, batch, num_activated_tokens]

    展平规则：
      - 按 C 顺序将除最后一维外的所有维度展平，保留最后一维长度 expected_last_dim；
      - 若两个数组同时存在，则先拼接 prefill，再拼接 decode（按上述顺序）。

    参数：
      - npz_path: .npz 文件路径；
      - as_list:  若为 True，则返回 List[List[...]]；否则返回 numpy.ndarray。
      - expected_last_dim: 末维期望大小，默认 8。

    返回：
      - numpy.ndarray，形状 [N, expected_last_dim]（默认）；或 List[List[...]]（as_list=True）。
    """

    def _reshape_to_rows(arr: np.ndarray, name: str) -> np.ndarray:
        if arr is None:
            return np.empty((0, expected_last_dim), dtype=np.float32)
        # 对象数组优先整体转换为数值数组（适用于标量对象数组）
        if arr.dtype == object:
            try:
                numeric = np.asarray(arr, dtype=np.int64)
                if numeric.ndim >= 1 and numeric.shape[-1] == expected_last_dim:
                    return numeric.reshape(-1, expected_last_dim)
                # 若最后一维不为 8，尝试按一维扁平后分组
                flat_num = numeric.reshape(-1)
                if flat_num.size % expected_last_dim != 0:
                    raise ValueError(
                        f"{name} 展平后长度 {flat_num.size} 不能被 {expected_last_dim} 整除，原形状 {numeric.shape}"
                    )
                return flat_num.reshape(-1, expected_last_dim)
            except Exception:
                # 回退：逐元素收集后分组为 8
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
        # 普通数值数组：优先按最后一维 8 处理
        if arr.ndim >= 1 and arr.shape[-1] == expected_last_dim:
            return arr.reshape(-1, expected_last_dim)
        # 兜底：按一维扁平后分组
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


# --- 3.2 从NPZ读取（保持原始形状），并断言末维 expected_last_dim ---
def load_npz_routing_keep_shape(npz_path: str, as_list: bool = False, expected_last_dim: int = 8):
    """
    读取 .npz 中的 'prefill' 与 'decode'，保持各自原始形状：
      - prefill 期望形状 [layer, iter, expected_last_dim]
      - decode  期望形状 [iter, layer, batch, expected_last_dim]

    若其中任意数组为对象数组（dtype=object），会尝试整体转换为数值数组。
    会断言两者的最后一维大小均为 expected_last_dim（默认 8）。

    返回：
      - 默认返回 numpy.ndarray 元组：(prefill, decode)
      - 若 as_list=True，返回 (prefill.tolist(), decode.tolist())
    """

    # 

    def _to_numeric_keep_shape(arr: np.ndarray, name: str) -> np.ndarray:
        if arr is None:
            return None
        if arr.dtype == object:
            # 整体转换为数值数组（适合对象数组内部为标量的情形）
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
    # ==================== 可配置区域 ====================
    simulation_config = {
        "iter": 128,
        "n_routed_experts": 256,
        "n_group": 8,
        "topk_group": 4,            # 选出多少个组
        "num_experts_per_tok": 8,   # 总共要选多少个专家（需能被 topk_group 整除）
        # 改动1：默认输出到 ../data/routing_outputs
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

        # 示例：转换为二维列表并展示前两个 token
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

        # 示例：直接从内存结果转换为二维列表并展示前两个 token
        expert_id_lists = extract_selected_experts_lists(routing_data)
        print("\n二维列表示例（前2个 token）：")
        for idx in range(min(2, len(expert_id_lists))):
            print(f"  token_{idx}: {expert_id_lists[idx]}")

def extract_prefill_and_decode_from_npz():
    npz_path = "../data/aime_ds_r1/moe_activations_batch0.npz"
    prefill_list, decode_list = load_npz_routing_flatten_last_dim(npz_path,as_list=False)
    print(prefill_list.shape)
    print(decode_list.shape)
    print(prefill_list[0])
    print(decode_list[0])
    return prefill_list, decode_list

def extract_prefill_and_decode_from_npz_as_list():
    # npz_path = "../data/aime_ds_r1/moe_activations_batch0.npz"
    npz_path = "../data/qwen_e128_a128/moe_activations_k128.npz"
    # 从 npz_path 提取出最后一个 / 之前，_a 之后的字符串
    # 如: "../data/qwen_e128_a128/moe_activations.npz" 提取 128


    

    prefill_list, decode_list = load_npz_routing_keep_shape(npz_path,as_list=False, expected_last_dim=128)
    print(prefill_list.shape)
    print(decode_list.shape)
    # print(prefill_list[0])
    print(decode_list[0,0,0])
    # 找到decode_list里最小的数
    # min_value = decode_list.min()
    # print("decode_list中的最小值:", min_value)
    # max_value = decode_list.max()
    # print("decode_list中的最大值:", max_value)
    return prefill_list, decode_list

# --- 4. 主执行逻辑 ---
if __name__ == "__main__":

    
    # example_gen_routing_list()
    # prefill_list, decode_list = extract_prefill_and_decode_from_npz()
    prefill_list, decode_list = extract_prefill_and_decode_from_npz_as_list()


