#need to have gfortran installed, then call 'make' to compile .so file, then this will successfully import
from h2o_pot import calc_hoh_pot
import numpy as np


def water_pot(cds):
    return calc_hoh_pot(cds, len(cds))


if __name__ == '__main__':
    x = np.random.random((100, 3, 3))
    v = water_pot(x)
    print(v)
