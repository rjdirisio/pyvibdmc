from pyvibdmc.simulation_utilities import *
import numpy as np


def hydrogen_harm(cds):
    """For testing only. Compulsory cds input"""
    massH = Constants.mass('H', to_AU=True)
    mass = massH
    omega = Constants.convert(3600., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)


def oh_stretch_harm(cds):
    """For tutorial. Compulsory numpy array cds input (N,1,1), output a numpy array in hartrees"""
    mass = Constants.reduced_mass('O-H', to_AU=True)
    omega = Constants.convert(3700., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)


def n2_stretch_harm(cds):
    """For tutorial. Compulsory numpy array cds input (N,1,1), output a numpy array in hartrees"""
    mass = Constants.reduced_mass('N-N', to_AU=True)
    omega = Constants.convert(2750., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)


def hcl_stretch_harm(cds):
    """For tutorial. Compulsory numpy array cds input (N,1,1), output a numpy array in hartrees"""
    mass = Constants.reduced_mass('H-Cl', to_AU=True)
    omega = Constants.convert(2850., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)
