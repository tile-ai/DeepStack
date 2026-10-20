"""Source implementation of the recovered finite capacity lookup."""
from pathlib import Path
from .recovered_capacity import CapacityLUT
_TABLE = CapacityLUT.load(Path(__file__).with_name('capacity_full_lut.bin'))
def p00(arch, noc):
    return _TABLE.lookup(arch, noc)
