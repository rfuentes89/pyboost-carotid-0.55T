"""pyboost -- BOOST carotid black-blood pulse sequence for PyPulseq at 0.55T.

Python/PyPulseq port of the KomaMRI reference simulation (``RR_sim.jl``) so the
BOOST (bright-blood + black-blood phase-sensitive) sequence can be exported as a
scanner-executable ``.seq`` file.

The physics of every preparation block mirrors the Koma reference; the imaging
readout adds real spatial encoding that the 1D contrast simulation never had.
"""

from .system import scanner_055T, FAT_PPM, fat_frequency
from .params import BoostParams
from .prep import fat_sat, t2_prep, adiabatic_ir
from .readout import bssfp_readout
from .boost import build_boost_sequence

__all__ = [
    "scanner_055T",
    "FAT_PPM",
    "fat_frequency",
    "BoostParams",
    "fat_sat",
    "t2_prep",
    "adiabatic_ir",
    "bssfp_readout",
    "build_boost_sequence",
]
