import torch
from typing import Optional, Sequence, Tuple, Union

class Tensor_Loc:
    def __init__(self, dtype: torch.dtype, loc: str, shape: Optional[Union[Sequence[int], torch.Size]] = None):
        self.dtype = dtype
        # self.num_bytes = int(torch.tensor([], dtype=dtype).element_size())
        self.num_bytes = torch.empty((), dtype=dtype).element_size()
        self.loc = loc  # 'ddr' | 'smem' | 'reg'
        self.shape: Optional[Tuple[int, ...]] = tuple(shape) if shape is not None else None
        # 


class OpBytes:
    def __init__(self, input1: Tensor_Loc, input2: Optional[Tensor_Loc], output: Tensor_Loc):
        self.input1 = input1
        self.input2 = input2
        self.output = output

    def to_mem_levels(self):
        def encode(io: Tensor_Loc):
            if io.loc == 'ddr':
                code = [1, 1, 1]
            elif io.loc == 'smem':
                code = [0, 1, 1]
            elif io.loc == 'reg':
                code = [0, 0, 1]
            else:
                raise ValueError("loc must be one of 'ddr', 'smem', 'reg'")
            return code + [io.num_bytes]

        result = {
            'in1': encode(self.input1),
        }
        if self.input2 is not None:
            result['in2'] = encode(self.input2)
        result['out1'] = encode(self.output)
        return result
    
    def get_dtype_bytes(self):
        input2_bytes = self.input2.num_bytes if self.input2 is not None else None
        return self.input1.num_bytes, input2_bytes, self.output.num_bytes

    def get_shapes(self) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
        """
        返回 (input1.shape, input2.shape 或 None, output.shape)。
        当 input2 不存在时，返回 None。
        """
        input2_shape = self.input2.shape if self.input2 is not None else None
        return self.input1.shape, input2_shape, self.output.shape



if __name__ == "__main__":
    # 简单示例测试
    # op = OpBytes(
    #     input1=Tensor_Loc(torch.float16, 'ddr'),
    #     input2=Tensor_Loc(torch.float16, 'smem'),
    #     output=Tensor_Loc(torch.float32, 'reg'),
    # )
    op = OpBytes(
        input1=Tensor_Loc(torch.bfloat16, 'ddr'),
        input2=Tensor_Loc(torch.float8_e4m3fn, 'smem'),
        output=Tensor_Loc(torch.float32, 'reg'),
    )
    levels = op.to_mem_levels()
    print(levels)

    # 期望输出：
    # {'in1': [1, 1, 1, 2], 'in2': [0, 1, 1, 2], 'out1': [0, 0, 1, 4]}
    # assert levels['in1'] == [1, 1, 1, 2]
    # assert levels['in2'] == [0, 1, 1, 2]
    # assert levels['out1'] == [0, 0, 1, 4]
    assert levels['in1'] == [1, 1, 1, 2]
    assert levels['in2'] == [0, 1, 1, 1]
    assert levels['out1'] == [0, 0, 1, 4]
    print("OpBytes mem_levels 测试通过")

    # 单输入算子测试（input2=None）
    op_single = OpBytes(
        input1=Tensor_Loc(torch.float16, 'ddr'),
        input2=None,
        output=Tensor_Loc(torch.float32, 'reg'),
    )
    levels_single = op_single.to_mem_levels()
    print(levels_single)
    assert 'in2' not in levels_single
    assert levels_single['in1'] == [1, 1, 1, 2]
    assert levels_single['out1'] == [0, 0, 1, 4]
    bytes_single = op_single.get_dtype_bytes()
    assert bytes_single == (2, None, 4)
    print("OpBytes 单输入测试通过")

    # 带 shape 的单输入算子测试
    # op_single_shape = OpBytes(
    #     input1=Tensor_Loc(torch.float16, 'ddr', shape=(32, 64)),
    #     input2=None,
    #     output=Tensor_Loc(torch.float32, 'reg', shape=(32, 64)),
    # )
    op_single_shape = OpBytes(
        input1=Tensor_Loc(torch.float16, 'ddr', [32, 64]),
        input2=None,
        output=Tensor_Loc(torch.float32, 'reg', [32, 64]),
    )
    # op_single_shape = OpBytes(
    #     input1=Tensor_Loc(torch.float16, 'ddr', [32, 64]),
    #     input2=Tensor_Loc(torch.float16, 'ddr', [64, 128]),
    #     output=Tensor_Loc(torch.float32, 'reg', [32, 128]),
    # )

    levels_single_shape = op_single_shape.to_mem_levels()
    print(levels_single_shape)
    assert 'in2' not in levels_single_shape
    assert levels_single_shape['in1'] == [1, 1, 1, 2]
    assert levels_single_shape['out1'] == [0, 0, 1, 4]
    bytes_single_shape = op_single_shape.get_dtype_bytes()
    assert bytes_single_shape == (2, None, 4)
    assert op_single_shape.input1.shape == (32, 64)
    assert op_single_shape.output.shape == (32, 64)
    print("OpBytes 带 shape 的单输入测试通过")
