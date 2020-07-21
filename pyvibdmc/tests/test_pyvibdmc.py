"""
Unit and regression test for the pyvibdmc package.
"""

# Import package, test suite, and other packages as needed
import numpy as np
from ..data import *

import pyvibdmc
import pytest
import sys

def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules

def test_initDmcObj():
    myDMC = pyvibdmc.DMC()
    assert isinstance(myDMC, pyvibdmc.DMC)