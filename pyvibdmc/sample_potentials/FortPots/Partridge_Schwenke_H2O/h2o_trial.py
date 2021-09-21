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


def deriv_angle(analyzer):
    hoh = analyzer.bond_angle(0, 2, 1)
    g = gmat()
    alpha = theta_freq / g
    x = hoh - theta_eq
    return (alpha / np.pi) ** 0.25 * (-alpha * x) * np.exp(-alpha * x ** 2 / 2)


def sderiv_angle(analyzer):
    hoh = analyzer.bond_angle(0, 2, 1)
    g = gmat()
    alpha = theta_freq / g
    x = hoh - theta_eq
    return (alpha / np.pi) ** 0.25 * (alpha ** 2 * x ** 2 - alpha) * np.exp(-alpha * x ** 2 / 2)


def trial_wavefunction(cds, ret_pdt=True):
    psi = np.zeros((len(cds), 3))
    analyzer = pv.AnalyzeWfn(cds)
    ohs = oh_dists(analyzer)
    for i in range(2):
        psi[:, i] = interpolate.splev(ohs[i], free_oh_wfn, der=0)
    psi[:, 2] = angle(analyzer)
    if ret_pdt:
        return np.prod(psi, axis=1)
    else:
        return psi


def first_deriv(cds):
    """computes dpsi/dr1, dpsi/dr2, dpsi/dtheta"""
    dpsi = np.zeros((len(cds), 3))
    analyzer = pv.AnalyzeWfn(cds)
    ohs = oh_dists(analyzer)
    for i in range(2):
        dpsi[:, i] = interpolate.splev(ohs[i], free_oh_wfn, der=1)
    dpsi[:, 2] = deriv_angle(analyzer)
    return dpsi.T


def sec_deriv(cds):
    """computes dpsi/dr1, dpsi/dr2, dpsi/dtheta"""
    sdpsi = np.zeros((len(cds), 3))
    analyzer = pv.AnalyzeWfn(cds)
    ohs = oh_dists(analyzer)
    for i in range(2):
        sdpsi[:, i] = interpolate.splev(ohs[i], free_oh_wfn, der=2)
    sdpsi[:, 2] = sderiv_angle(analyzer)
    return sdpsi.T


def dpsi_dx(cds):
    """Retruns the first and second derivative"""
    trl = trial_wavefunction(cds, ret_pdt=False)
    dpsi_dr = first_deriv(cds) / trl.T
    dr_dx = pv.ChainRuleHelper.dr_dx(cds, [[0, 2], [1, 2]])
    dth_dx = pv.ChainRuleHelper.dth_dx(cds, [[0, 2, 1]])
    dint_dx = np.concatenate([dr_dx, dth_dx])
    dp_dx = pv.ChainRuleHelper.dpsidx(dpsi_dr, dint_dx)
    d2psi_dr2 = sec_deriv(cds) / trl.T
    d2r_dx2 = pv.ChainRuleHelper.d2r_dx2(cds, [[0, 2], [1, 2]], dr_dx)
    d2th_dx2 = pv.ChainRuleHelper.d2th_dx2(cds, [[0, 2, 1]])
    d2int_dx2 = np.concatenate([d2r_dx2, d2th_dx2])
    d2p_dx2 = pv.ChainRuleHelper.d2psidx2(d2psi_dr2, d2int_dx2, dpsi_dr, dint_dx)
    return dp_dx, d2p_dx2
