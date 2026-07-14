"""Tests for the MRzero-differentiable flip optimization (pyboost.diffopt).

These exercise the real MRzero engine, so they are slower than the pure-pypulseq
tests; they are skipped if MRzeroCore/torch are unavailable.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mr0 = pytest.importorskip("MRzeroCore")

from pyboost import (BoostParams, scanner_055T, import_mra_for_optimization,
                     differentiable_flip_signal)
from pyboost.phantom import TISSUE_PROPERTIES


@pytest.fixture(scope="module")
def imported():
    p = BoostParams(nx=8, ny=8, im_segments=8, dummy_heart_beats=1, centric=True,
                    t2prep_duration=0.06, im_flip_angle=(90.0, 80.0))
    seq0, img_idx = import_mra_for_optimization(p, scanner_055T(), base_flip_deg=90.0)
    return seq0, img_idx, p


def test_imaging_pulses_identified(imported):
    seq0, img_idx, p = imported
    # ~im_segments imaging pulses per heartbeat, over (dummy + n_shots) heartbeats.
    n_hb = p.dummy_heart_beats + p.n_shots
    assert len(img_idx) == pytest.approx(p.im_segments * n_hb, abs=2)
    # The isolated T2-prep 90 deg pulses must NOT be selected.
    angles = np.array([float(seq0[i].pulse.angle) * 180 / np.pi for i in img_idx])
    assert np.all(np.abs(angles - 90.0) < 1.0)


def test_signal_is_differentiable(imported):
    seq0, img_idx, p = imported
    per = p.im_segments * p.nx
    tp = TISSUE_PROPERTIES["blood"]
    obj = mr0.CustomVoxelPhantom(pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"],
                                 T2=tp["T2"], T2dash=0.03, D=0.0, voxel_size=5e-3)
    flip = torch.tensor(70.0, requires_grad=True)
    s = differentiable_flip_signal(seq0, img_idx, flip, obj, per)
    s.backward()
    assert flip.grad is not None and torch.isfinite(flip.grad)
    assert abs(float(flip.grad)) > 0  # signal actually depends on the flip
