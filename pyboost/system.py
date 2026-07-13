"""Scanner / system limits for the 0.55T BOOST sequence.

Mirrors the KomaMRI reference scanner (``RR_sim.jl:7-10``)::

    sys = Scanner(); sys.B0 = 0.55; sys.Gmax = 40e-3; sys.Smax = 25e-3

The Koma contrast simulation does not slew-limit its hand-built spoilers, so its
``Gmax = 40e-3`` / ``Smax = 25e-3`` are effectively decorative there. A *real*
executable bSSFP at TR=7 ms / TE=3.51 ms will not close its timing at 25 T/m/s
(the slice-select ramp alone eats ~0.75 ms). The defaults below therefore match
a Siemens MAGNETOM Free.Max (a common 0.55T system): ~26 mT/m and ~45 T/m/s.
Override ``max_grad`` / ``max_slew`` for your actual scanner.
"""

from __future__ import annotations

import pypulseq as pp

# Fat-water chemical shift. Same value used by the Koma reference
# (``RR_sim.jl:54``). The *frequency* offset scales with B0, so it is computed
# from the system gamma in :func:`fat_frequency`.
FAT_PPM: float = -3.4e-6


def scanner_055T(
    max_grad: float = 26.0,
    max_slew: float = 45.0,
    grad_unit: str = "mT/m",
    slew_unit: str = "T/m/s",
    rf_dead_time: float = 100e-6,
    rf_ringdown_time: float = 60e-6,
    adc_dead_time: float = 10e-6,
    b0: float = 0.55,
) -> pp.Opts:
    """Return a :class:`pypulseq.Opts` configured for 0.55T.

    Parameters
    ----------
    max_grad, max_slew, grad_unit, slew_unit
        Gradient amplitude / slew-rate limits and their units. Defaults mirror
        the numbers in the Koma reference but should be set to your scanner.
    rf_dead_time, rf_ringdown_time, adc_dead_time
        Hardware dead/ringdown times enforced by ``check_timing``. The RF
        ringdown is a touch longer than the 1.5T norm because low-field bodies
        run cooler and vendors often relax it; keep the value your scanner
        reports.
    b0
        Main field strength in tesla.
    """
    return pp.Opts(
        max_grad=max_grad,
        grad_unit=grad_unit,
        max_slew=max_slew,
        slew_unit=slew_unit,
        rf_dead_time=rf_dead_time,
        rf_ringdown_time=rf_ringdown_time,
        adc_dead_time=adc_dead_time,
        B0=b0,
    )


def fat_frequency(system: pp.Opts, fat_ppm: float = FAT_PPM) -> float:
    """Fat resonance offset [Hz] at the system field strength.

    ``Δf = fat_ppm * B0 * gamma`` -> about -80 Hz at 0.55T versus about -220 Hz
    at 1.5T. The narrow low-field separation is exactly why fat saturation is
    harder here and why the FatSat pulse must be long and its bandwidth tight.
    """
    return fat_ppm * system.B0 * system.gamma
