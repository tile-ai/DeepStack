from mosaic.llm_arch import DeepSeekV3, LLM_Arch, Qwen3_235b_a22b, Qwen3_480b_a35b, Llama3_70b, Llama3_405b


def task_combination_2():
    # model_arch = task_combination[0]
    # BS = task_combination[1]
    # INPUT_SEQ = task_combination[2]
    # TASK_MAX_SEQ = task_combination[3]
    # for parallel decoding
    # PARALLEL_SEQ = task_combination[4]
    task_combinations = []

    PARALLEL_DECODING = 1
    for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
    # for model_arch in [DeepSeekV3(),Llama3_70b(), Llama3_405b()]:
        for BS in [1, 16, 128, 1024]:
            for INPUT_SEQ in [128, 1024, 8192]:
                for TASK_MAX_SEQ in [256, 8192, 64*1024]:
                    
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    
    return task_combinations

def task_combination_3():
    # model_arch = task_combination[0]
    # BS = task_combination[1]
    # INPUT_SEQ = task_combination[2]
    # TASK_MAX_SEQ = task_combination[3]
    # for parallel decoding
    # PARALLEL_SEQ = task_combination[4]
    task_combinations = []

    PARALLEL_DECODING = 1
    for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b()]:
        for BS in [1, 16, 128, 1024]:
            for INPUT_SEQ in [128, 1024, 8192]:
                for TASK_MAX_SEQ in [256, 8192, 64*1024]:
                    PARALLEL_DECODING = INPUT_SEQ
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    
    return task_combinations


def task_combination_4():
    # model_arch = task_combination[0]
    # BS = task_combination[1]
    # INPUT_SEQ = task_combination[2]
    # TASK_MAX_SEQ = task_combination[3]
    # for parallel decoding
    # PARALLEL_SEQ = task_combination[4]
    task_combinations = []

    for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
    # for model_arch in [ Llama3_70b(), Llama3_405b()]:
        # for BS in [1, 16, 128, 1024]:
        for BS in [1, 4, 8, 16,64, 128,256, 1024]:
            # for INPUT_SEQ in [128, 1024, 8192, 32*1024]:
            for INPUT_SEQ in [1024]:
                    TASK_MAX_SEQ = INPUT_SEQ
                    PARALLEL_DECODING = INPUT_SEQ
                    task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))

    return task_combinations

def customized_prefill_task_combination():
    # model_arch = task_combination[0]
    # BS = task_combination[1]
    # INPUT_SEQ = task_combination[2]
    # TASK_MAX_SEQ = task_combination[3]
    # for parallel decoding
    # PARALLEL_SEQ = task_combination[4]
    task_combinations = []

    for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
    # for model_arch in [ Llama3_70b(), Llama3_405b()]:
        # for BS in [1, 16, 128, 1024]:
        for BS in [1, 4, 8, 16,64, 128,256, 1024]:
            # for INPUT_SEQ in [128, 1024, 8192, 32*1024]:
            for INPUT_SEQ in [1024]:
                    TASK_MAX_SEQ = INPUT_SEQ
                    PARALLEL_DECODING = INPUT_SEQ
                    task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    
    # additional for small model llama3_70b
    for model_arch in [Llama3_70b()]:
        for BS in [2048, 4096, 8192]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))

    return task_combinations

def customized_decoding_task_combination():
    # model_arch = task_combination[0]
    # BS = task_combination[1]
    # INPUT_SEQ = task_combination[2]
    # TASK_MAX_SEQ = task_combination[3]
    # for parallel decoding
    # PARALLEL_SEQ = task_combination[4]
    task_combinations = []

    # PARALLEL_DECODING = 1
    # for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
    # for model_arch in [DeepSeekV3(),Llama3_70b(), Llama3_405b()]:
    #     for BS in [1, 16, 128, 1024]:
    #         for INPUT_SEQ in [128, 1024, 8192]:
    #             for TASK_MAX_SEQ in [256, 8192, 64*1024]:
                    
    #                 if TASK_MAX_SEQ > INPUT_SEQ:
    #                     task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    # task_combinations = [
    #     (Qwen3_235b_a22b(), 1024, 1024, 8192, 1),
    #     (Qwen3_235b_a22b(), 1024, 1024, 64*1024, 1),
    # ]



    PARALLEL_DECODING = 1
    for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
    # for model_arch in [DeepSeekV3(),Llama3_70b(), Llama3_405b()]:
        # for BS in [1, 16, 128, 1024]:
        for BS in [1, 4, 16, 64, 128, 256, 512, 1024]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    # additional for small model llama3_70b
    for model_arch in [Llama3_70b()]:
        for BS in [2048, 4096, 8192]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))

    return task_combinations

def scale_64_512_nodes_task_combination_decode():
    task_combinations = []
    PARALLEL_DECODING = 1
    for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
        for BS in [1, 4, 16, 64, 128, 256, 1024]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    return task_combinations

def scale_64_512_nodes_task_combination_prefill():
    task_combinations = []
    for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
        for BS in [1, 4, 16, 64, 128, 256, 1024]:
            # for INPUT_SEQ in [128, 1024, 8192, 32*1024]:
            for INPUT_SEQ in [1024]:
                    for TASK_MAX_SEQ in [2048]:
                        PARALLEL_DECODING = INPUT_SEQ
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))

    return task_combinations

def dpsk_decode_task_combination():
    task_combinations = []
    PARALLEL_DECODING = 1
    for model_arch in [DeepSeekV3()]:
        for BS in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    return task_combinations

def dpsk_decode_max_stps_task_combination():
    task_combinations = []
    PARALLEL_DECODING = 1
    for model_arch in [DeepSeekV3()]:
        for BS in [512, 1024, 2048, 4096, 8192]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    return task_combinations

def qwen3_235b_a22b_decode_task_combination():
    task_combinations = []
    PARALLEL_DECODING = 1
    for model_arch in [Qwen3_235b_a22b()]:
        for BS in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    return task_combinations

def llama3_70b_decode_task_combination():
    task_combinations = []
    PARALLEL_DECODING = 1
    for model_arch in [Llama3_70b()]:
        for BS in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    return task_combinations

def llama3_405b_decode_task_combination():
    task_combinations = []
    PARALLEL_DECODING = 1
    for model_arch in [Llama3_405b()]:
        for BS in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    return task_combinations

def hybrid_decode_task_combination_0301():
    task_combinations = []
    PARALLEL_DECODING = 1
    # for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
    for model_arch in [DeepSeekV3()]:
        for BS in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            for INPUT_SEQ in [1024]:
                for TASK_MAX_SEQ in [2048]:
                    if TASK_MAX_SEQ > INPUT_SEQ:
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    return task_combinations

def hybrid_prefill_task_combination_0301():
    # task_combinations = []
    # PARALLEL_DECODING = 1
    # # for model_arch in [DeepSeekV3(), Qwen3_235b_a22b(), Llama3_70b(), Llama3_405b()]:
    # for model_arch in [DeepSeekV3(), Llama3_70b()]:
    #     for BS in [512]:
    #         for INPUT_SEQ in [1024]:
    #             for TASK_MAX_SEQ in [2048]:
    #                 if TASK_MAX_SEQ > INPUT_SEQ:
    #                     task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    task_combinations = []
    for model_arch in [DeepSeekV3(), Llama3_70b(),]:
        for BS in [128]:
            # for INPUT_SEQ in [128, 1024, 8192, 32*1024]:
            for INPUT_SEQ in [4096]:
                    for TASK_MAX_SEQ in [2048]:
                        PARALLEL_DECODING = INPUT_SEQ
                        task_combinations.append((model_arch, BS, INPUT_SEQ, TASK_MAX_SEQ, PARALLEL_DECODING))
    return task_combinations