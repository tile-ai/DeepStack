import torch

def get_bytes(dtype: torch.dtype):
    return torch.empty((), dtype=dtype).element_size()

if __name__ == "__main__":
    print(get_bytes(torch.float16))
    print(get_bytes(torch.float32))
    print(get_bytes(torch.float64))
    print(get_bytes(torch.int8))
    print(get_bytes(torch.int16))
    print(get_bytes(torch.int32))
    print(get_bytes(torch.int64))
    print(get_bytes(torch.bool))
    print(get_bytes(torch.float8_e4m3fn))
    print(get_bytes(torch.float8_e5m2))
    print(get_bytes(torch.float8_e4m3fn))
    print(get_bytes(torch.float4_e2m1fn_x2))