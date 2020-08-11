from ...simulation_utilities import *
def HODMC(cds,atmStr):
    """Compulsory cds, atmStr input"""
    if len(atmStr) == 2:
        masses = [Constants.mass(x, to_AU=True) for x in atmStr]
    elif len(atmStr) == 1 or type(atmStr)==str:
        mass = Constants.mass(*atmStr, to_AU=True)
    else:
        raise Exception
    omega = Constants.convert(3000., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)