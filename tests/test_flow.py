"""Tests for the approximate flow (complete-washout) model in pyboost.flow."""

import numpy as np
import pytest

mr0 = pytest.importorskip("MRzeroCore")

from pyboost import (BoostParams, scanner_055T, build_boost_sequence,
                     compartment_maps, split_phantoms, locate_imaging_reps,
                     locate_prep_pulses, strip_preparation)
from pyboost.phantom import carotid_phantom_maps, CarotidGeometry, LABELS


@pytest.fixture(scope="module")
def maps():
    return carotid_phantom_maps(CarotidGeometry(fov=0.06, matrix=32))


@pytest.fixture(scope="module")
def imported():
    p = BoostParams(nx=8, ny=8, im_segments=8, dummy_heart_beats=0, centric=True,
                    inversion_kind="block", ir_inversion_time=0.30)
    seq = build_boost_sequence(p, scanner_055T(), add_trigger=False)
    seq.write("/tmp/test_flow.seq")
    return mr0.Sequence.import_file("/tmp/test_flow.seq"), p


def test_compartments_partition_the_phantom(maps):
    """Static + blood compartments cover every non-air voxel, without overlap."""
    static = [n for n in LABELS if n not in ("air", "blood")]
    pd_static = compartment_maps(maps, static)["PD"]
    pd_blood = compartment_maps(maps, ["blood"])["PD"]
    assert np.all(pd_static * pd_blood == 0)                 # disjoint
    np.testing.assert_allclose(pd_static + pd_blood, maps["PD"])
    assert (pd_blood > 0).sum() == (maps["label"] == LABELS["blood"]).sum()


def test_split_phantoms_voxel_counts(maps):
    static_obj, blood_obj = split_phantoms(maps, 0.06)
    n_blood = int((maps["label"] == LABELS["blood"]).sum())
    n_static = int(((maps["PD"] > 0) & (maps["label"] != LABELS["blood"])).sum())
    assert len(blood_obj.PD) == n_blood
    assert len(static_obj.PD) == n_static


def test_prep_pulses_are_the_preparation_only(imported):
    seq0, p = imported
    angles = np.array([float(r.pulse.angle) * 180 / np.pi for r in seq0])
    prep = locate_prep_pulses(seq0, p)
    imaging = locate_imaging_reps(seq0, p)
    assert set(prep).isdisjoint(imaging)
    assert len(prep) + len(imaging) == len(seq0)
    # T2-prep (90/180/90), the inversion and the FatSat are all >= 90 deg...
    assert np.all(angles[prep] >= 89.0)
    # ...and every iNAV ramp start (3.2 deg) is classified as imaging, never prep.
    ramp_starts = np.where(np.abs(angles - p.inav_flip_angle) < 0.5)[0]
    assert len(ramp_starts) > 0
    assert set(ramp_starts).issubset(imaging)


def test_strip_preparation_zeroes_prep_and_keeps_timing(imported):
    seq0, p = imported
    fresh = mr0.Sequence.import_file("/tmp/test_flow.seq")
    prep = strip_preparation(fresh, p)
    assert prep, "no preparation pulses found"
    for i in prep:
        assert float(fresh[i].pulse.angle) == pytest.approx(0.0, abs=1e-6)
    # Imaging pulses untouched and, crucially, timing/gradients preserved so the
    # stripped copy shares the k-space trajectory (signals can be added).
    for i in locate_imaging_reps(fresh, p):
        assert float(fresh[i].pulse.angle) == pytest.approx(
            float(seq0[i].pulse.angle), abs=1e-6)
    for i in range(len(fresh)):
        np.testing.assert_allclose(fresh[i].event_time.detach().numpy(),
                                   seq0[i].event_time.detach().numpy(), rtol=1e-6)
        np.testing.assert_allclose(fresh[i].gradm.detach().numpy(),
                                   seq0[i].gradm.detach().numpy(), rtol=1e-6)


def test_stripped_blood_stays_at_equilibrium(imported):
    """With the preparation stripped, blood reaches the readout unprepared.

    Its first-line signal must exceed the prepared case (T2-prep + inversion can
    only reduce the available magnetization) -- this is the inflow effect.
    """
    seq0, p = imported
    tp = dict(PD=0.7, T1=1.122, T2=0.263)
    obj = mr0.CustomVoxelPhantom(pos=[[0.0, 0.0, 0.0]], PD=tp["PD"], T1=tp["T1"],
                                 T2=tp["T2"], T2dash=0.03, D=0.0, voxel_size=5e-3)
    prepared, _ = mr0.util.simulate(seq0, obj)
    fresh_seq = mr0.Sequence.import_file("/tmp/test_flow.seq")
    strip_preparation(fresh_seq, p)
    fresh, _ = mr0.util.simulate(fresh_seq, obj)
    n = p.nx                                            # first acquired line
    assert abs(fresh.reshape(-1)[:n]).mean() > abs(prepared.reshape(-1)[:n]).mean()
