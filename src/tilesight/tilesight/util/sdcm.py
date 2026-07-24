from scipy.stats import norm
from scipy.special import comb
import numpy as np
import math

def norm_cdf_approx_5(x):
    """
    An approximation of the cumulative distribution function for the standard normal distribution.
    """
    # Constants in the rational approximation
    a1, a2, a3, a4, a5, a6 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429, 2.506628274
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    
    # Calculate approximation
    approx = 1.0 - (1.0 / (math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)) * \
             (a1 * k + a2 * k**2 + a3 * k**3 + a4 * k**4 + a5 * k**5)

    # a1, a2, a3 = 0.4361836, 0.1201676, 0.937298
    # k = 1.0 / (1.0 + 0.33267 * abs(x))
    
    # # Calculate approximation
    # approx = 1.0 - (1.0 / (math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)) * \
    #          (a1 * k + a2 * k**2 + a3 * k**3 )

    return approx if x >= 0 else 1 - approx

def norm_cdf_approx_3(x):

    a1, a2, a3 = 0.4361836, 0.1201676, 0.937298
    k = 1.0 / (1.0 + 0.33267 * abs(x))
    
    # Calculate approximation
    approx = 1.0 - (1.0 / (math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)) * \
             (a1 * k + a2 * k**2 + a3 * k**3 )

    return approx if x >= 0 else 1 - approx

def sdcm(D, A, B):
    if (B==0):
        return 0
    # Probability of success
    p = A / B

    # Mean and standard deviation of the approximating normal distribution
    mu = D * p
    sigma = math.sqrt(D * p * (1 - p))

    prob = 0
    D = math.ceil(D)
    
    if D <= A - 1:
        prob = 1
    elif D >= 1e8:
        prob = 0
    elif D <= 8:
        for a in range(A):
            prob += comb(D, a) * (A/B)**a * ((B-A)/B)**(D-a)
    else:
        norm_input=(A - 1 + 0.5 - mu) / sigma
        # prob = norm.cdf((A - 1 + 0.5 - mu) / sigma)
        # prob = norm_cdf_approx_5(norm_input)
        prob = norm_cdf_approx_3(norm_input)
        # 
        # prob = 0.5
    
    return prob

# # 示例用法：
# D = 100   # 示例参数，应根据实际情况进行调整
# A = 8    # 示例参数，应根据实际情况进行调整
# B = 128    # 示例参数，应根据实际情况进行调整
# probability = sdcm(D, A, B)
# print(probability)
