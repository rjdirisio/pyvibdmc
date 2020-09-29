from pyvibdmc.simulation_utilities import *
import numpy as np

def HODMC(cds):
    """Compulsory cds, atmStr input"""
    massH = Constants.mass('H',to_AU=True)
    mass = massH
    omega = Constants.convert(3600., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)
