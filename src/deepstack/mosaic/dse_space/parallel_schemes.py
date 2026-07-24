# from mosaic.parallelism import ParallelScheme
# from sympy import factorint

# def parallel_dse(parallel: ParallelScheme, instance_num: int) -> ParallelScheme:

#     # tp, ep, sp, cp, dp, pp, fsdp
#     # fsdp is an option for dp, not a factor of instance_num
#     # instance_num = tp * ep * sp * cp * dp * pp 
#     # tp, ep, sp, cp, dp, pp = factorize(instance_num)

#     # print all the factors
#     factors = factorint(instance_num)
#     print(factors)
#     expanded = [p for p, exp in factors.items() for _ in range(exp)]
#     print(expanded)





#     return parallel

# if __name__ == "__main__":
#     parallel = ParallelScheme(tp=2, ep=8, sp=4, cp=1, dp=2, pp=2, fsdp=False)
#     instance_num = 128
#     parallel = parallel_dse(parallel, instance_num)
#     print(parallel)

from typing import Dict, List, Tuple
from itertools import combinations, product
from sympy import factorint
from mosaic.parallelism import ParallelScheme

Labels = Tuple[str, str, str, str, str, str]
DEFAULT_LABELS: Labels = ("tp", "ep", "sp", "cp", "dp", "pp")

def _compositions(n: int, k: int):
    """生成所有 k 元非负整数组合，和为 n（stars and bars）。"""
    if k <= 0:
        return
    if n == 0:
        yield (0,) * k
        return
    total = n + k - 1
    for cuts in combinations(range(total), k - 1):
        prev = -1
        parts = []
        for c in cuts + (total,):
            parts.append(c - prev - 1)
            prev = c
        yield tuple(parts)

def all_parallel_schemes_list(instance_num: int,
                              labels: Labels = DEFAULT_LABELS) -> List[Dict[str, int]]:
    """
    返回所有 (tp, ep, sp, cp, dp, pp) 方案的列表，乘积等于 instance_num。
    - 仅用 sympy.factorint
    - 即使多个槽得到相同数值（如都为 2 的幂），也不会出现重复结果
    """
    if instance_num < 1:
        raise ValueError("instance_num 必须是正整数")
    if len(labels) != 6:
        raise ValueError("需要 6 个标签（tp, ep, sp, cp, dp, pp）")

    prime_powers = factorint(instance_num)  # {p: e}
    if not prime_powers:
        return [{label: 1 for label in labels}]

    k = len(labels)
    per_prime_splits = []
    primes = []
    for p, e in prime_powers.items():
        primes.append(p)
        per_prime_splits.append(list(_compositions(e, k)))

    out: List[Dict[str, int]] = []
    for splits_product in product(*per_prime_splits):
        slot_vals = [1] * k
        for prime_idx, split_vec in enumerate(splits_product):
            p = primes[prime_idx]
            for j, exp in enumerate(split_vec):
                if exp:
                    slot_vals[j] *= p ** exp
        out.append(dict(zip(labels, slot_vals)))
    return out

def all_parallel_schemes(instance_num: int,
                         labels: Labels = DEFAULT_LABELS) -> List[ParallelScheme]:
    """
    返回所有 ParallelScheme(tp, ep, sp, cp, dp, pp, fsdp=...) 的组合。
    每个方案会生成两个版本：fsdp=False 和 fsdp=True。
    """
    if instance_num < 1:
        raise ValueError("instance_num 必须是正整数")
    if len(labels) != 6:
        raise ValueError("需要 6 个标签（tp, ep, sp, cp, dp, pp）")

    prime_powers = factorint(instance_num)  # {p: e}
    if not prime_powers:
        # instance_num == 1 的情况
        return [
            ParallelScheme(tp=1, ep=1, sp=1, cp=1, dp=1, pp=1, fsdp=f)
            for f in (False, True)
        ]

    k = len(labels)
    per_prime_splits = []
    primes = []
    for p, e in prime_powers.items():
        primes.append(p)
        per_prime_splits.append(list(_compositions(e, k)))

    schemes: List[ParallelScheme] = []
    for splits_product in product(*per_prime_splits):
        slot_vals = [1] * k
        for prime_idx, split_vec in enumerate(splits_product):
            p = primes[prime_idx]
            for j, exp in enumerate(split_vec):
                if exp:
                    slot_vals[j] *= p ** exp

        # 对每个组合生成 fsdp=True / False 两个版本
        kwargs = dict(zip(labels, slot_vals))
        for fsdp_value in (False, True):
            schemes.append(ParallelScheme(**kwargs, fsdp=fsdp_value))

    return schemes


def filter_parallel_schemes_by_max(schemes: List[ParallelScheme],
                                   label: str,
                                   max_value: int,
                                   *,
                                   valid_labels: Labels = DEFAULT_LABELS) -> List[ParallelScheme]:
    """
    过滤 ParallelScheme 列表，使给定维度 `label` 的取值 <= `max_value`。

    - `label` 必须是 {tp, ep, sp, cp, dp, pp} 之一
    - `max_value` 必须是正整数
    """
    if label not in valid_labels:
        raise ValueError(f"label 必须是 {valid_labels} 之一, 但得到: {label}")
    if max_value < 1:
        raise ValueError("max_value 必须是正整数")

    return [s for s in schemes if getattr(s, label) <= max_value]


def filter_parallel_schemes_by_product_max(schemes: List[ParallelScheme],
                                           label1: str,
                                           label2: str,
                                           max_product: int,
                                           *,
                                           valid_labels: Labels = DEFAULT_LABELS) -> List[ParallelScheme]:
    """
    过滤 ParallelScheme 列表，使给定两个维度 `label1` 与 `label2` 的乘积 <= `max_product`。

    - `label1` 与 `label2` 必须是 {tp, ep, sp, cp, dp, pp} 之一
    - `max_product` 必须是正整数
    """
    if label1 not in valid_labels:
        raise ValueError(f"label1 必须是 {valid_labels} 之一, 但得到: {label1}")
    if label2 not in valid_labels:
        raise ValueError(f"label2 必须是 {valid_labels} 之一, 但得到: {label2}")
    if max_product < 1:
        raise ValueError("max_product 必须是正整数")

    return [
        s for s in schemes
        if getattr(s, label1) * getattr(s, label2) <= max_product
    ]


def filter_illegal_fsdp(schemes: List[ParallelScheme]) -> List[ParallelScheme]:
    """
    过滤非法的 FSDP 组合：当 dp == 1 时，仅保留 fsdp == False 的方案；
    其它情况下（dp != 1）保持不变。
    """
    return [s for s in schemes if not (s.dp == 1 and s.fsdp is True)]

def gen_correlation_parallel_schemes(max_product: int):
    #  qwen, bs 128, tp 1, ep 32, dp 2, pp 4
    tp_base_scheme = ParallelScheme(tp=1, ep=32, sp=1, cp=1, dp=2, pp=4, fsdp=False)
    # qwen, bs 128, tp 8, ep 1, dp 2, pp 16
    ep_base_scheme = ParallelScheme(tp=8, ep=1, sp=1, cp=1, dp=2, pp=16, fsdp=False)
    #  tp 2, ep 16, dp 2, pp 4
    all_base_scheme = ParallelScheme(tp=2, ep=16, sp=1, cp=1, dp=2, pp=4, fsdp=False)

    # 
    scheme_list=[]

    # gen tp correlation schemes
    tp_scheme_list=[]
    current_tp = tp_base_scheme.tp
    while True:
        # 计算当前并行度的乘积
        product = current_tp * tp_base_scheme.ep * tp_base_scheme.sp * tp_base_scheme.cp * tp_base_scheme.dp * tp_base_scheme.pp
        if product > max_product:
            break
        # 追加新的scheme
        tp_scheme_list.append(ParallelScheme(
            tp=current_tp,
            ep=tp_base_scheme.ep,
            sp=tp_base_scheme.sp,
            cp=tp_base_scheme.cp,
            dp=tp_base_scheme.dp,
            pp=tp_base_scheme.pp,
            fsdp=tp_base_scheme.fsdp
        ))
        current_tp *= 2
    scheme_list.extend(tp_scheme_list)

    # gen ep correlation schemes
    ep_scheme_list=[]
    current_ep = ep_base_scheme.ep
    while True:
        # 计算当前并行度的乘积
        product = current_ep * ep_base_scheme.tp * ep_base_scheme.sp * ep_base_scheme.cp * ep_base_scheme.dp * ep_base_scheme.pp
        if product > max_product:
            break
        # 追加新的scheme
        ep_scheme_list.append(ParallelScheme(
            tp=ep_base_scheme.tp,
            ep=current_ep,
            sp=ep_base_scheme.sp,
            cp=ep_base_scheme.cp,
            dp=ep_base_scheme.dp,
            pp=ep_base_scheme.pp,
            fsdp=ep_base_scheme.fsdp
        ))
        current_ep *= 2
    scheme_list.extend(ep_scheme_list)

    # gen sp correlation schemes
    sp_scheme_list = []
    current_sp = all_base_scheme.sp
    while True:
        # 计算当前并行度的乘积
        product = (
            all_base_scheme.tp
            * all_base_scheme.ep
            * current_sp
            * all_base_scheme.cp
            * all_base_scheme.dp
            * all_base_scheme.pp
        )
        if product > max_product:
            break
        sp_scheme_list.append(
            ParallelScheme(
                tp=all_base_scheme.tp,
                ep=all_base_scheme.ep,
                sp=current_sp,
                cp=all_base_scheme.cp,
                dp=all_base_scheme.dp,
                pp=all_base_scheme.pp,
                fsdp=all_base_scheme.fsdp,
            )
        )
        current_sp *= 2
    scheme_list.extend(sp_scheme_list)

    # gen cp correlation schemes
    cp_scheme_list = []
    current_cp = all_base_scheme.cp
    while True:
        # 计算当前并行度的乘积
        product = (
            all_base_scheme.tp
            * all_base_scheme.ep
            * all_base_scheme.sp
            * current_cp
            * all_base_scheme.dp
            * all_base_scheme.pp
        )
        if product > max_product:
            break
        cp_scheme_list.append(
            ParallelScheme(
                tp=all_base_scheme.tp,
                ep=all_base_scheme.ep,
                sp=all_base_scheme.sp,
                cp=current_cp,
                dp=all_base_scheme.dp,
                pp=all_base_scheme.pp,
                fsdp=all_base_scheme.fsdp,
            )
        )
        current_cp *= 2
    scheme_list.extend(cp_scheme_list)

    # gen dp correlation schemes
    dp_scheme_list = []
    current_dp = all_base_scheme.dp
    while True:
        # 计算当前并行度的乘积
        product = (
            all_base_scheme.tp
            * all_base_scheme.ep
            * all_base_scheme.sp
            * all_base_scheme.cp
            * current_dp
            * all_base_scheme.pp
        )
        if product > max_product:
            break
        dp_scheme_list.append(
            ParallelScheme(
                tp=all_base_scheme.tp,
                ep=all_base_scheme.ep,
                sp=all_base_scheme.sp,
                cp=all_base_scheme.cp,
                dp=current_dp,
                pp=all_base_scheme.pp,
                fsdp=all_base_scheme.fsdp,
            )
        )
        current_dp *= 2
    scheme_list.extend(dp_scheme_list)

    # gen pp correlation schemes
    pp_scheme_list = []
    current_pp = all_base_scheme.pp
    while True:
        # 计算当前并行度的乘积
        product = (
            all_base_scheme.tp
            * all_base_scheme.ep
            * all_base_scheme.sp
            * all_base_scheme.cp
            * all_base_scheme.dp
            * current_pp
        )
        if product > max_product:
            break
        pp_scheme_list.append(
            ParallelScheme(
                tp=all_base_scheme.tp,
                ep=all_base_scheme.ep,
                sp=all_base_scheme.sp,
                cp=all_base_scheme.cp,
                dp=all_base_scheme.dp,
                pp=current_pp,
                fsdp=all_base_scheme.fsdp,
            )
        )
        current_pp *= 2
    scheme_list.extend(pp_scheme_list)

    # for scheme in scheme_list:
    #     if scheme.world_size() <= 32:
    #         print(scheme)

    return scheme_list

if __name__ == "__main__":
    # schemes = all_parallel_schemes(64)   # 256 = 2^6
    # print(len(schemes))  # 462 , 462 * 2 for fsdp=False and fsdp=True
    
    # schemes = all_parallel_schemes(256)   # 256 = 2^8
    # print(len(schemes))  # 1287 , 1287 * 2 for fsdp=False and fsdp=True

    # schemes_1024 = all_parallel_schemes(1024)  # 1024 = 2^10
    # print(len(schemes_1024))  # 3003 , 3003 * 2 for fsdp=False and fsdp=True
    
    # print(1*2*3*4*5*6) # 720
    # schemes = all_parallel_schemes(1*2*3*4*5*6)  
    # print(len(schemes)) # 15876 , 15876 * 2 for fsdp=False and fsdp=True

    schemes = all_parallel_schemes(256)
    print(len(schemes)) # 1287 , 1287 * 2 for fsdp=False and fsdp=True, 2574
    # for scheme in schemes:
    #     print(scheme)
    schemes = filter_parallel_schemes_by_max(schemes, "sp", 1)
    print(len(schemes)) # 990

    # schemes = filter_parallel_schemes_by_max(schemes, "pp", 16)
    # print(len(schemes)) # 920

    schemes = filter_parallel_schemes_by_product_max(schemes, "dp", "pp", 16)
    print(len(schemes)) # 720

    schemes = filter_illegal_fsdp(schemes)
    print(len(schemes)) # 575

    schemes = gen_correlation_parallel_schemes(4096)


    # not oom schemes:
    # for ds, 825 
    # 

    
