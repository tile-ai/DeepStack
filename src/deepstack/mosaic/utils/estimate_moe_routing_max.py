# # Estimator for E[max EP-bin] for one experiment (R=1)
# # Using the formula: E[max S_b] ≈ μ + σ * a_m,
# # where μ = T * 8/EP, σ depends on routing, and a_m ≈ sqrt(2 ln m) - (ln ln m + ln(4π)) / (2 sqrt(2 ln m)),
# # with m ≈ EP - 1 to account for equicorrelation and the sum-to-constant constraint.

# import math
# import pprint


# def a_m_refined(m: float) -> float:
#     L = math.log(m)
#     return math.sqrt(2*L) - (math.log(L) + math.log(4*math.pi)) / (2*math.sqrt(2*L))

# def qwen_sigma(M, n, EP, T):
#     p = 1/EP
#     var_per_token = n * p * (1 - p) * ((M - n) / (M - 1))
#     return math.sqrt(T * var_per_token)

# def deepseek_sigma_ep8_aligned(T):
#     return math.sqrt(T)  # EP=8 对齐特例：单 token 方差=1

# def estimate_moe_routing_max(M, n, EP, T, R, method="qwen"):
#     mu = T * n / EP
#     if EP == 1:
#         return mu
#     if method == "qwen":
#         sigma = qwen_sigma(M, n, EP, T)
#     elif method == "deepseek":
#         sigma = deepseek_sigma_ep8_aligned(T)  # assumes EP=8 aligned
#     m_eff = (EP - 1) * R
#     am = a_m_refined(m_eff)
#     # return {"mu": mu, "sigma": sigma, "m_eff": m_eff, "a_m": am, "E_max_est": mu + sigma * am}
#     return mu + sigma * am

# def estimate_moe_routing_imbalance_overhead(M, n, EP, T, R, method="qwen"):
#     if EP == 1:
#         return 1
#     mu = T * n / EP
#     if method == "qwen":
#         sigma = qwen_sigma(M, n, EP, T)
#     elif method == "deepseek":
#         sigma = deepseek_sigma_ep8_aligned(T)  # assumes EP=8 aligned
#     m_eff = (EP - 1) * R
#     am = a_m_refined(m_eff)
#     # return {"mu": mu, "sigma": sigma, "m_eff": m_eff, "a_m": am, "E_max_est": mu + sigma * am}
#     # return mu + sigma * am
#     return 1 + sigma * am / mu


# if __name__ == "__main__":\
#     # Given parameters
#     M = 256
#     n = 8
#     EP = 8
#     R = 4
#     T = 32

#     out = {
#         "inputs": {"M":M, "n":n, "EP":EP, "R":R, "T":T},
#         "Qwen_MoE": estimate_moe_routing_max(M, n, EP, T, R, method="qwen"),
#         "DeepSeek": estimate_moe_routing_max(M, n, EP, T, R, method="deepseek"),
#         "Qwen_MoE_Unbalanced_Overhead": estimate_moe_routing_imbalance_overhead(M, n, EP, T, R, method="qwen"),
#         "DeepSeek_Unbalanced_Overhead": estimate_moe_routing_imbalance_overhead(M, n, EP, T, R, method="deepseek")
#     }

#     print(out)

import math
import pprint

def a_m_refined(m: float) -> float:
    """
    计算 m 个标准正态分布变量期望最大值的 refined 近似值。
    添加了 m <= 1 的边界检查来防止 math domain error。
    """
    # m_eff = (EP - 1) * R
    # 如果 m <= 1 (例如 EP=2, R=1 导致 m=1)，
    # 此时 log(m) <= 0，会导致 log(L) 失败。
    # 对于 m=1，最大值的期望就是其均值，对于标准正态分布即为 0。
    if m <= 1.0:
        return 0.0

    L = math.log(m)
    # 既然 m > 1, 那么 L > 0, math.log(L) 和 sqrt(L) 都是安全的。
    
    sqrt_2L = math.sqrt(2 * L)
    
    # 分母 (2 * sqrt_2L) 也不可能为 0
    return sqrt_2L - (math.log(L) + math.log(4 * math.pi)) / (2 * sqrt_2L)

def qwen_sigma(M, n, EP, T):
    """计算 Qwen-MoE (超几何分布) 的 sigma"""
    # 鲁棒性：EP 必须 >= 1。如果 EP <= 0，p 会无效或导致除零。
    # 主函数中已检查 EP=1，这里假设 EP >= 2。
    if EP < 2:
        return 0.0
        
    p = 1 / EP
    
    # 鲁棒性：处理 (M-1) 除零的情况
    # (M-n)/(M-1) 是有限群体校正因子
    # 如果 M=1, 那么 n 必须=1，方差为 0
    # 如果 M=n, 方差也为 0 (抽了所有)
    if M <= 1 or M == n:
        var_per_token = 0.0
    else:
        finite_pop_correction = (M - n) / (M - 1)
        var_per_token = n * p * (1 - p) * finite_pop_correction
        
    return math.sqrt(T * var_per_token)

def deepseek_sigma_ep8_aligned(T):
    # EP=8 对齐特例：单 token 方差=1
    # 鲁棒性：确保 T 不是负数
    return math.sqrt(max(0.0, T))

def estimate_moe_routing_max(M, n, EP, T, R, method="qwen"):
    # 鲁棒性：处理 EP <= 0 的无效输入
    if EP <= 0:
        print("Warning: EP must be >= 1. Returning 0.")
        return 0.0

    mu = T * n / EP
    
    if EP == 1:
        return mu
        
    if method == "qwen":
        sigma = qwen_sigma(M, n, EP, T)
    elif method == "deepseek":
        if EP != 8:
            print(f"Warning: Using deepseek_sigma_ep8_aligned, but EP is {EP} (not 8).")
        sigma = deepseek_sigma_ep8_aligned(T)
    else:
        raise ValueError(f"Unknown method: {method}")

    m_eff = (EP - 1) * R
    am = a_m_refined(m_eff)
    
    return mu + sigma * am

def estimate_moe_routing_imbalance_overhead(M, n, EP, T, R, method="qwen"):
    # 鲁棒性：处理 EP <= 0 的无效输入
    if EP <= 0:
        print("Warning: EP must be >= 1. Returning 1.0 (no overhead).")
        return 1.0

    if EP == 1:
        return 1.0  # 单专家没有不平衡开销
        
    mu = T * n / EP
    
    # 鲁棒性：防止除以零
    # 如果 T=0 或 n=0, 那么 mu=0。此时开销无意义，返回 1.0
    if mu == 0.0:
        return 1.0
        
    if method == "qwen":
        sigma = qwen_sigma(M, n, EP, T)
    elif method == "deepseek":
        if EP != 8:
            print(f"Warning: Using deepseek_sigma_ep8_aligned, but EP is {EP} (not 8).")
        sigma = deepseek_sigma_ep8_aligned(T)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    m_eff = (EP - 1) * R
    am = a_m_refined(m_eff)
    
    # mu 已经在上面检查过不为 0
    return 1.0 + sigma * am / mu


if __name__ == "__main__":
    # 原始参数 (不会出错)
    M = 256
    n = 8
    EP = 8
    R = 4
    T = 32

    out = {
        "inputs": {"M":M, "n":n, "EP":EP, "R":R, "T":T},
        "Qwen_MoE": estimate_moe_routing_max(M, n, EP, T, R, method="qwen"),
        "DeepSeek": estimate_moe_routing_max(M, n, EP, T, R, method="deepseek"),
        "Qwen_MoE_Unbalanced_Overhead": estimate_moe_routing_imbalance_overhead(M, n, EP, T, R, method="qwen"),
        "DeepSeek_Unbalanced_Overhead": estimate_moe_routing_imbalance_overhead(M, n, EP, T, R, method="deepseek")
    }

    print("--- 原始参数测试 ---")
    pprint.pprint(out)

    # 导致错误的参数 (EP=2, R=1)
    M_err = 256
    n_err = 8
    EP_err = 2
    R_err = 1
    T_err = 32

    out_err = {
        "inputs": {"M":M_err, "n":n_err, "EP":EP_err, "R":R_err, "T":T_err},
        "Qwen_MoE": estimate_moe_routing_max(M_err, n_err, EP_err, T_err, R_err, method="qwen"),
        "DeepSeek": estimate_moe_routing_max(M_err, n_err, EP_err, T_err, R_err, method="deepseek"),
        "Qwen_MoE_Unbalanced_Overhead": estimate_moe_routing_imbalance_overhead(M_err, n_err, EP_err, T_err, R_err, method="qwen"),
        "DeepSeek_Unbalanced_Overhead": estimate_moe_routing_imbalance_overhead(M_err, n_err, EP_err, T_err, R_err, method="deepseek")
    }
    
    print("\n--- 边界条件测试 (EP=2, R=1) ---")
    pprint.pprint(out_err)
    
    # 另一个边界条件 (n=0 导致 mu=0)
    out_zero_mu = {
        "inputs": {"M":M, "n":0, "EP":EP, "R":R, "T":T},
        "Qwen_MoE_Overhead_n0": estimate_moe_routing_imbalance_overhead(M, 0, EP, T, R, method="qwen"),
        "DeepSeek_Overhead_n0": estimate_moe_routing_imbalance_overhead(M, 0, EP, T, R, method="deepseek")
    }
    print("\n--- 边界条件测试 (n=0) ---")
    pprint.pprint(out_zero_mu)