def find_max_power_of_two(max_const, N):

    '''
    Args:
        max_const: 最大值
        N: 输入值

    Returns:
        小于等于N的最大的2的倍数
    '''
    
    # 找到小于等于N的最大的2的倍数
    assert N > 0

    max_power_of_two = 1

    while max_power_of_two * 2 <= min(max_const, N):
        max_power_of_two *= 2
    # 已经确保 max_power_of_two <= min(max_const, N)
    return max_power_of_two