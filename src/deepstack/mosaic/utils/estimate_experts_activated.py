def estimate_experts_activated(num_tokens_per_device: int, num_experts_per_device: int):
    # num_tokens_per_device / num_experts_per_device 往下整除是多少？
    few_activated_times = num_tokens_per_device // num_experts_per_device
    more_activated_times = few_activated_times + 1

    num_more_activated_experts = num_tokens_per_device - few_activated_times * num_experts_per_device
    num_few_activated_experts = num_experts_per_device - num_more_activated_experts

    # return num_device_few, few_experts, num_device_more, more_experts
    # return num_experts_1, num_activated_times_1, num_experts_2, num_activated_times_2
    return num_few_activated_experts, few_activated_times, num_more_activated_experts, more_activated_times


if __name__ == "__main__":
    num_tokens_per_device = 100
    num_experts_per_device = 32
    num_few_activated_experts, few_activated_times, num_more_activated_experts, more_activated_times = estimate_experts_activated(num_tokens_per_device, num_experts_per_device)
    print(f"num_few_activated_experts: {num_few_activated_experts}, few_activated_times: {few_activated_times}, num_more_activated_experts: {num_more_activated_experts}, more_activated_times: {more_activated_times}")