import numpy as np
from ..data import *

def HODMC(cds,atmStr):
    if len(atmStr) == 2:
        masses = [Constants.mass(x, to_AU=True) for x in atmStr]
    elif len(atmStr) == 1 or type(atmStr)==str:
        mass = Constants.mass(*atmStr, to_AU=True)

    omega = Constants.convert(3000., 'wavenumbers', to_AU=True)
    return np.squeeze(0.5 * mass * omega ** 2 * cds ** 2)

HODMC(np.zeros(100),'H')