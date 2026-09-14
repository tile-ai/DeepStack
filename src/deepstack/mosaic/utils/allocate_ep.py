import logging
log = logging.getLogger(__name__) 
import math
from mosaic.parallelism import ParallelScheme



# def allocate_ep(bs: int, seq: int, parallel: ParallelScheme):
def allocate_ep(parallel: ParallelScheme, bs: int, seq: int):

    def find_proper_factors(n: int) -> list:
        factors = []
        divisor = 2
        while n > 1 and divisor < 100:
            while n % divisor == 0:
                factors.append(divisor)
                n //= divisor
            divisor += 1
        # If no factor is found and n>1, n itself is prime; add it as a factor
        if not factors and n > 1:
            factors.append(n)
        return factors
        
    # allocate ep for input activations

    shard_bs = math.ceil(bs / parallel.dp)
    shard_seq = math.ceil(seq / parallel.sp)

    target_ep = parallel.ep
    if target_ep <= 1:
        log.info("allocate_ep -> target_ep=%s, ep1=%s, ep2=%s", target_ep, 1, 1)
        parallel.ep1 = 1
        parallel.ep2 = 1
        return parallel
    factors = find_proper_factors(target_ep)

    ep1 = 1
    ep2 = 1

    # Try assigning each prime factor of ep in order: prioritize seq -> ep2, then bs -> ep1
    for f in factors:
        if shard_seq % f == 0:
            shard_seq //= f
            ep2 *= f
        elif shard_bs % f == 0:
            shard_bs //= f
            ep1 *= f
        else:
            # This factor divides neither the remaining shard_seq nor shard_bs; skip it
            continue

    log.info("allocate_ep -> target_ep=%s, ep1=%s, ep2=%s", target_ep, ep1, ep2)
    parallel.ep1 = ep1
    parallel.ep2 = ep2
    return parallel
    # return ep1, ep2


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # seq = 9
    # bs = 8
    seq = 2
    bs = 3
    # parallel = ParallelScheme(tp=2, ep=8, sp=1, cp=1, dp=2, pp=2,fsdp=False)
    parallel = ParallelScheme(tp=2, ep=8, sp=1, cp=1, dp=1, pp=2,fsdp=False)
    # ep1, ep2 = allocate_ep(bs, seq, parallel)
    # parallel = allocate_ep(parallel, bs, seq)
    allocate_ep(parallel, bs, seq)

    log.info("shard_bs: %s, shard_seq: %s", math.ceil(bs / parallel.dp), math.ceil(seq / parallel.sp))
    log.info("ep1: %s, ep2: %s", parallel.ep1, parallel.ep2)