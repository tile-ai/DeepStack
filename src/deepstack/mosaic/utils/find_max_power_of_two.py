def find_max_power_of_two(max_const, N):

    """Args:
        max_const: Maximum value.
        N: Input value.

    Returns:
        The largest multiple of 2 less than or equal to N.
    """
    
    assert N > 0

    max_power_of_two = 1

    while max_power_of_two * 2 <= min(max_const, N):
        max_power_of_two *= 2
    return max_power_of_two