from pyvibdmc.simulation_utilities import *
import numpy as np

def oh_stretch_morse(disp):
    """Params From DMC book chapter"""
    mass = Constants.reduced_mass('O-H', to_AU=True)
    omega = Constants.convert(3704.5, 'wavenumbers', to_AU=True)
    omega_x = Constants.convert(75.3,'wavenumbers',to_AU=True)
    De = (omega ** 2 / (4 * omega_x))
    alpha = np.sqrt(mass * (omega ** 2.) / 2. / De)
    pot_vals = De * np.square((1 - np.exp(-alpha * disp)))
    return pot_vals.squeeze()
