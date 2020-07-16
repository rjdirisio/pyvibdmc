"""
Unit and regression test for the pyvibdmc package.
"""

# Import package, test suite, and other packages as needed
import pyvibdmc
import pytest
import sys

def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules
