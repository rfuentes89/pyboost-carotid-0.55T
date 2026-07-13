"""Tests for the concentric carotid phantom (pyboost.phantom)."""

import numpy as np
import pytest

from pyboost.phantom import (carotid_phantom_maps, CarotidGeometry, LABELS,
                             TISSUE_PROPERTIES)


@pytest.fixture
def maps():
    return carotid_phantom_maps(CarotidGeometry())


def test_all_tissues_present(maps):
    lab = maps["label"]
    for name in ("blood", "wall", "muscle", "fat"):
        assert np.any(lab == LABELS[name]), f"{name} missing from phantom"


def test_lumen_inside_wall(maps):
    """Every lumen pixel is surrounded by wall/lumen -- concentric geometry."""
    lab = maps["label"]
    blood = lab == LABELS["blood"]
    # Dilate the blood mask by one pixel; the new ring must be wall or blood.
    ys, xs = np.where(blood)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ring = lab[np.clip(ys + dy, 0, lab.shape[0] - 1),
                   np.clip(xs + dx, 0, lab.shape[1] - 1)]
        assert np.all(np.isin(ring, [LABELS["blood"], LABELS["wall"]]))


def test_relaxation_maps_match_labels(maps):
    lab = maps["label"]
    for name, props in TISSUE_PROPERTIES.items():
        m = lab == LABELS[name]
        assert np.allclose(maps["T1"][m], props["T1"])
        assert np.allclose(maps["T2"][m], props["T2"])
        assert np.allclose(maps["PD"][m], props["PD"])


def test_air_has_zero_pd(maps):
    assert np.all(maps["PD"][maps["label"] == LABELS["air"]] == 0.0)


def test_fat_offset_default_on_resonance(maps):
    # Default keeps fat on-resonance so FatSat behaves (see phantom.py note).
    assert np.all(maps["B0"] == 0.0)


def test_fat_offset_opt_in():
    g = CarotidGeometry(apply_fat_offset=True)
    m = carotid_phantom_maps(g)
    fat = m["label"] == LABELS["fat"]
    assert np.allclose(m["B0"][fat], g.fat_offset_hz())
    assert g.fat_offset_hz() == pytest.approx(-79.6, abs=1.0)


def test_to_mrzero_voxel_count(maps):
    from pyboost.phantom import to_mrzero_phantom
    obj = to_mrzero_phantom(maps, CarotidGeometry().fov)
    assert len(obj.PD) == int((maps["PD"] > 0).sum())
