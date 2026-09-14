import logging

log = logging.getLogger(__name__)

import math
def find_waves_duplicate_factor(waves: float, wave_util_ratio = 0.7):

    duplicate_factor = 1
    
    log.info ("before shrinking, waves: %s, wave_util_ratio: %s",  waves, (waves*duplicate_factor)/math.ceil(waves*duplicate_factor))

    # duplicate_factor should be power of 2
    # while (math.ceil(waves*duplicate_factor)/(waves*duplicate_factor)) < wave_util_ratio and duplicate_factor <= 8:
    while ((waves*duplicate_factor)/math.ceil(waves*duplicate_factor)) < wave_util_ratio and duplicate_factor <= 8:
        # log.info ("duplicate_factor: %s, waves: %s, wave_util_ratio: %s", duplicate_factor, waves, wave_util_ratio)
        # print ("duplicate_factor: %s, waves: %s, wave_util_ratio: %s", duplicate_factor, waves, wave_util_ratio)
        duplicate_factor *=2
    
    log.info("after shrinking, duplicate_factor: %s, waves: %s, wave_util_ratio: %s", duplicate_factor, waves*duplicate_factor, (waves*duplicate_factor)/math.ceil(waves*duplicate_factor))
    

    return duplicate_factor

def find_proper_factors(n: int) -> list:
    factors = []
    divisor = 2
    while n > 1 and divisor < 100:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    if not factors:
        factors.append(n)
    return factors

def shrink_tiling(tb_shape:list, duplicate_factor:int):
    # first we will find the factor of duplicate_factor


    proper_factors = find_proper_factors(duplicate_factor)

    new_shape = list(tb_shape)

    # traverse factors in reverse order
    for factor in reversed(proper_factors):
        # pick the largest dimension that is divisible by current factor
        candidate_indices = [idx for idx, dim in enumerate(new_shape) if isinstance(dim, int) and dim % factor == 0]
        if not candidate_indices:
            continue
        best_idx = max(candidate_indices, key=lambda i: new_shape[i])
        new_shape[best_idx] //= factor

    return new_shape

def shrink_tiling_by_waves(waves: float, tb_shape: list, wave_util_ratio: float = 0.7):
    """Compute duplicate_factor from waves and wave_util_ratio, then shrink tb_shape.

    Args:
        waves: Number of waves used to compute duplicate_factor.
        tb_shape: Original tiling shape.
        wave_util_ratio: Target wave utilization; defaults to 0.7.

    Returns:
        The reduced tiling shape as a list.
    """
    duplicate_factor = find_waves_duplicate_factor(waves, wave_util_ratio)
    return shrink_tiling(tb_shape, duplicate_factor)

if __name__ == "__main__":
    print(find_proper_factors(99))
    print(shrink_tiling([12, 256], 3))

    print(shrink_tiling([12, 256], 96))

    print(shrink_tiling_by_waves(waves= 0.5714285714285714, tb_shape= [8, 256]))

    print(shrink_tiling_by_waves(waves= 0.5714285714285714, tb_shape= [1, 1]))


