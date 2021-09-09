import pyvibdmc as pv
import numpy as np
from scipy import interpolate

r1_eq = pv.Constants.convert(0.95784, 'angstroms', to_AU=True)
r2_eq = pv.Constants.convert(0.95783997, 'angstroms', to_AU=True)
theta_eq = np.deg2rad(104.5080029)
theta_freq = pv.Constants.convert(1668.4590610594878, 'wavenumbers', to_AU=True)
inv_mh = 1 / pv.Constants.mass('H')
inv_mo = 1 / pv.Constants.mass('O')
wvfn = np.load("free_oh_wvfn.npy")
free_oh_wfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)

def oh_dists(analyzer):
    oh1 = analyzer.bond_length(0, 2)
    oh2 = analyzer.bond_length(1, 2)
    return [oh1, oh2]


def gmat():
    return inv_mh / r1_eq ** 2 + inv_mh / r2_eq ** 2 + inv_mo * \
           (1 / r1_eq ** 2 + 1 / r2_eq ** 2 - 2 * np.cos(theta_eq) / (r1_eq * r2_eq))


def angle(analyzer):
    hoh = analyzer.bond_angle(0, 2, 1)
    g = gmat()
    alpha = theta_freq / g
    return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (hoh - theta_eq) ** 2 / 2)


def trial_wavefunction(cds):
    psi = np.zeros((len(cds), 3))
    analyzer = pv.AnalyzeWfn(cds)
    ohs = oh_dists(analyzer)
    for i in range(2):
        psi[:, i] = interpolate.splev(ohs[i], free_oh_wfn, der=0)
    psi[:, 2] = angle(analyzer)
    return np.prod(psi, axis=1)
