from ...simulation_utilities import *
def HODMC(cds):
    """Compulsory cds, atmStr input"""
    massH = Constants.mass('H',to_AU=True)
    mass = massH
    # mass = (massH*massO)/(massH+massO) #reduced mass of OH stretch
    omega = Constants.convert(3600., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)