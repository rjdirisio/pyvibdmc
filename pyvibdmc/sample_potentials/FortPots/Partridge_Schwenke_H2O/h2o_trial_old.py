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
#grd = wvfn[:,0]
free_oh_wfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)

#dense_grd = np.linspace(grd.min(),grd.max(), num = 5000)
#print(dense_grd[1]-dense_grd[0])
#wfnn = interpolate.splev(dense_grd, free_oh_wfn, der=1)
#der = interpolate.splev(dense_grd, free_oh_wfn, der=1)
#sder = interpolate.splev(dense_grd, free_oh_wfn, der=1)
#big_bops = np.stack((dense_grd,wfnn,der,sder))
#print(big_bops.shape)
#np.save("free_oh_wvfn_dense.npy",big_bops)

def get_angle(cds):
    analyzer = pv.AnalyzeWfn(cds)
    hoh = analyzer.bond_angle(0, 2, 1)
    return hoh

def get_cos_angle(cds):
    analyzer = pv.AnalyzeWfn(cds)
    hoh = analyzer.bond_angle(0, 2, 1)
    return np.cos(hoh)

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
    psi = np.zeros((3, len(cds)))
    analyzer = pv.AnalyzeWfn(cds)
    ohs = oh_dists(analyzer)
    for i in range(2):
        psi[i] = interpolate.splev(ohs[i], free_oh_wfn, der=0)
    psi[2] = angle(analyzer)
    if ret_pdt:
        return np.prod(psi, axis=0)
    else:
        return psi


def first_deriv(cds):
    """computes dpsi/dr1, dpsi/dr2, dpsi/dtheta"""
    dpsi = np.zeros((3,len(cds)))
    analyzer = pv.AnalyzeWfn(cds)
    ohs = oh_dists(analyzer)
    for i in range(2):
        dpsi[i] = interpolate.splev(ohs[i], free_oh_wfn, der=1)
    dpsi[2] = deriv_angle(analyzer)
    return dpsi


def sec_deriv(cds):
    """computes dpsi/dr1, dpsi/dr2, dpsi/dtheta"""
    sdpsi = np.zeros((3,len(cds)))
    analyzer = pv.AnalyzeWfn(cds)
    ohs = oh_dists(analyzer)
    for i in range(2):
        sdpsi[i] = interpolate.splev(ohs[i], free_oh_wfn, der=2)
    sdpsi[2] = sderiv_angle(analyzer)
    return sdpsi


def dpsi_dx(cds):
    """Retruns the first and second derivative"""
    # First, calculate trial wave function (num_walkers x num_modes)
    trl = trial_wavefunction(cds, ret_pdt=False)
    dpsi_dr = first_deriv(cds) / trl
    d2psi_dr2 = sec_deriv(cds) / trl
    ohs = [[0, 2], [2, 1]]
    hoh = [0, 2, 1]

    # Chain rule derivatives
    crh = pv.ChainRuleHelper(cds, np)

    dr_dxs = np.stack([crh.dr_dx(oh) for oh in ohs])
    d2r_dx2s = np.stack([crh.d2r_dx2(oh, dr_dx=dr_dxs[num]) for num, oh in enumerate(ohs)])

    dcth_dx = crh.dcth_dx(hoh,
                          dr_da=dr_dxs[0],
                          dr_dc=dr_dxs[1])
    dth_dx = crh.dth_dx(hoh,
                        dcth_dx=dcth_dx,
                        dr_da=dr_dxs[0],
                        dr_dc=dr_dxs[1])

    d2th_dx2 = crh.d2th_dx2(hoh,
                            dcth_dx=dcth_dx,
                            dr_da=dr_dxs[0],
                            dr_dc=dr_dxs[1],
                            d2r_da2=d2r_dx2s[0],
                            d2r_dc2=d2r_dx2s[1])
    # Calculate dpsi/dx / psi
    dint_dx = np.concatenate((dr_dxs, np.expand_dims(dth_dx, 0)))
    dp_dx = crh.dpsidx(dpsi_dr, dint_dx)

    # Calculate d2psi/dx2 / psi
    d2int_dx2 = np.concatenate((d2r_dx2s, np.expand_dims(d2th_dx2, 0)))
    d2p_dx2 = crh.d2psidx2(d2psi_dr2, d2int_dx2, dpsi_dr, dint_dx)
    return dp_dx, d2p_dx2
