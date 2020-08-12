import numpy as np
def dummyPot(cds):
    """Assumes cds = numWalkers x numAtoms x 3 np.ndarray"""
    return np.zeros(cds.shape[0])
