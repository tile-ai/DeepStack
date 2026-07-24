import os
import numpy as np
from typing import Sequence, Optional


def _sample_unique_indices(rng: np.random.Generator, num_experts: int, k: int) -> np.ndarray:
    """
    从 [0, num_experts) 中无放回采样 k 个索引，返回升序的 int64 数组。
    """
    if k > num_experts:
        raise ValueError(f"k={k} 不能大于 num_experts={num_experts}")
    picked = rng.choice(num_experts, size=k, replace=False)
    picked.sort()
    return picked.astype(np.int64, copy=False)


def generate_random_routing_npz(
    output_path: str,
    num_experts: int = 128,
    num_layers: int = 16,
    num_iters: int = 256,
    batch_size: int = 4,
    k: int = 8,
    seed: Optional[int] = 42,
):
    """
    生成一个包含 'prefill' 与 'decode' 的 .npz：
      - prefill 形状为 [num_layers, num_iters, k]
      - decode  形状为 [num_iters, num_layers, batch_size, k]
    每个位置存放从 [0, num_experts) 无放回采样得到的 k 个 expert ID（升序）。
    """
    rng = np.random.default_rng(seed)

    # prefill: [layer, iter, k]
    prefill = np.empty((num_layers, num_iters, k), dtype=np.int64)
    for l in range(num_layers):
        for it in range(num_iters):
            prefill[l, it] = _sample_unique_indices(rng, num_experts, k)

    # decode: [iter, layer, batch, k]
    decode = np.empty((num_iters, num_layers, batch_size, k), dtype=np.int64)
    for it in range(num_iters):
        for l in range(num_layers):
            for b in range(batch_size):
                decode[it, l, b] = _sample_unique_indices(rng, num_experts, k)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, prefill=prefill, decode=decode)


def generate_all_ks(
    base_dir: str,
    ks: Sequence[int] = (1, 8, 32, 128),
    num_experts: int = 128,
    num_layers: int = 16,
    num_iters: int = 256,
    batch_size: int = 4,
    seed: Optional[int] = 42,
):
    """
    批量为多个 k 生成 .npz 文件，文件命名为 moe_activations_k{K}.npz。
    输出目录按 data/qwen_e{num_experts}_a{k}/ 区分不同 k。
    """
    for k in ks:
        output_dir_k = os.path.join(base_dir, f"qwen_e{num_experts}_a{k}")
        os.makedirs(output_dir_k, exist_ok=True)
        outfile = os.path.join(output_dir_k, f"moe_activations_k{k}.npz")
        generate_random_routing_npz(
            output_path=outfile,
            num_experts=num_experts,
            num_layers=num_layers,
            num_iters=num_iters,
            batch_size=batch_size,
            k=k,
            seed=None if seed is None else (seed + k),  # 让不同 k 的文件也有不同随机性
        )
        print(f"已生成: {outfile}")


if __name__ == "__main__":
    # 目标目录与参数
    # 相对项目根路径：DeepStack data/
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    base_dir = os.path.abspath(base_dir)

    generate_all_ks(
        base_dir=base_dir,
        ks=(1, 8, 32, 128),
        num_experts=128,
        num_layers=1,
        num_iters=4096,
        batch_size=1,
        seed=42,
    )
    print("全部 npz 生成完成。基目录：", base_dir)
