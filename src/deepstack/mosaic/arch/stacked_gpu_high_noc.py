from .stacked_gpu_base import stacked_gpu_base


class stacked_gpu_high_noc(stacked_gpu_base):
    __slots__ = ()

    def __init__(self):
        super().__init__(_profile_name="high_noc")
