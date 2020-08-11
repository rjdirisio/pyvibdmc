import pyvibdmc
import pytest
import sys
sim_ex_dir = "exSimResults"
def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules
