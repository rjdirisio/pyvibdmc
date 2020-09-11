import pyvibdmc
from ..analysis import *
import numpy as np
import pytest
import sys

sim_ex_dir = "exSimResults"


def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules


def test_ana_wfn_blens():
    expected_blens = np.array([1.81005599, 1.81505599, 1.82005599, 1.82505599, 1.83005599, 1.83505599,
                               1.84005599, 1.84505599, 1.85005599, 1.85505599])
    water = np.array([[1.81005599, 0., 0.],
                      [-0.45344658, 1.75233806, 0.],
                      [0., 0., 0.]])
    sample_waters = np.broadcast_to(water, (10,) + np.shape(water)).copy() #make ten copies

    for wn, water in enumerate(sample_waters):
        water[0, 0] += 0.005 * wn

    ana_o = AnalyzeWfn(sample_waters)
    bond_lengths = ana_o.bond_length(0, 2)  # OH bond length
    print(bond_lengths)
    assert np.allclose(bond_lengths,expected_blens)
