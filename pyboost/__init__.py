"""pyboost -- BOOST carotid black-blood pulse sequence for PyPulseq at 0.55T.

Python/PyPulseq port of the KomaMRI reference simulation (``RR_sim.jl``) so the
BOOST (bright-blood + black-blood phase-sensitive) sequence can be exported as a
scanner-executable ``.seq`` file.

The physics of every preparation block mirrors the Koma reference; the imaging
readout adds real spatial encoding that the 1D contrast simulation never had.
"""

from .system import scanner_055T, FAT_PPM, fat_frequency
from .params import BoostParams
from .prep import fat_sat, t2_prep, inversion
from .readout import bssfp_readout
from .boost import build_boost_sequence
from .mra import build_mra_sequence
from .diffopt import import_mra_for_optimization, differentiable_flip_signal

__all__ = [
    "scanner_055T",
    "FAT_PPM",
    "fat_frequency",
    "BoostParams",
    "fat_sat",
    "t2_prep",
    "inversion",
    "bssfp_readout",
    "build_boost_sequence",
    "build_mra_sequence",
    "import_mra_for_optimization",
    "differentiable_flip_signal",
]
