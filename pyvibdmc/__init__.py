"""
PyVibDMC
A general purpose diffusion monte carlo code for studying vibrational problems
"""

# Add imports here
from .pyvibdmc import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
