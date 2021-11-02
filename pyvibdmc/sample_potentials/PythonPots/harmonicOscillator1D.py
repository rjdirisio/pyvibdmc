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

def oh_stretch_harm_shifted(cds):
    """For tutorial. Compulsory numpy array cds input (N,1,1), output a numpy array in hartrees"""
    mass = Constants.reduced_mass('O-H', to_AU=True)
    omega = Constants.convert(3700., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * (cds-Constants.convert(0.98,'angstroms',to_AU=True)) ** 2)

def oh_stretch_harm_with_arg(cds,extra_args):
    """For tutorial. Compulsory numpy array cds input (N,1,1), output a numpy array in hartrees"""
    mass = extra_args['mass']
    omega = extra_args['freq']
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)

def oh_stretch_harm_loadtxt(cds):
    """Will try to load random.txt. Part of testing that Potential_NoMP works the way it is intended"""
    mass = Constants.reduced_mass('O-H', to_AU=True)
    omega = Constants.convert(3700., 'wavenumbers', to_AU=True)
    a = np.loadtxt('random.txt')
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
