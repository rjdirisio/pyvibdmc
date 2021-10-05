import pyvibdmc as pv
import numpy as np
import cupy as xp
from scipy import interpolate

"""GPU version of this trial wfn + derivatives"""

r1_eq = pv.Constants.convert(0.95784, 'angstroms', to_AU=True)
r2_eq = pv.Constants.convert(0.95783997, 'angstroms', to_AU=True)
theta_eq = np.deg2rad(104.5080029)
theta_freq = pv.Constants.convert(1668.4590610594878, 'wavenumbers', to_AU=True)
inv_mh = 1 / pv.Constants.mass('H')
inv_mo = 1 / pv.Constants.mass('O')

# load in prepared, dense grid wave function
wvfn = xp.load("free_oh_wvfn_dense.npy")
gridpts = wvfn[0]
wfn = wvfn[1]
der_wfn = wvfn[2]
sder_wfn = wvfn[3]


def dot_pdt(v1, v2):
    new_v1 = xp.expand_dims(v1, axis=1)
    new_v2 = xp.expand_dims(v2, axis=2)
    return xp.matmul(new_v1, new_v2).squeeze()


def bond_angle(cds):
    vec1 = cds[:, 0] - cds[:, 2]
    vec2 = cds[:, 1] - cds[:, 2]
    dotV = xp.arccos(dot_pdt(vec1, vec2) /
                     (xp.linalg.norm(vec1, axis=1) * xp.linalg.norm(vec2, axis=1)))
    return dotV


def oh_dists(cds):
    oh1 = xp.linalg.norm(cds[:, 0] - cds[:, 2], axis=1)
    oh2 = xp.linalg.norm(cds[:, 1] - cds[:, 2], axis=1)
    return xp.array((oh1, oh2))


def gmat():
    return inv_mh / r1_eq ** 2 + inv_mh / r2_eq ** 2 + inv_mo * \
           (1 / r1_eq ** 2 + 1 / r2_eq ** 2 - 2 * np.cos(theta_eq) / (r1_eq * r2_eq))


def angle(cds, hoh):
    # hoh = bond_angle(cds)
    g = gmat()
    alpha = theta_freq / g
    return (alpha / np.pi) ** (1 / 4) * xp.exp(-alpha * (hoh - theta_eq) ** 2 / 2)


def deriv_angle(cds, hoh):
    # hoh = bond_angle(cds)
    g = gmat()
    alpha = theta_freq / g
    x = hoh - theta_eq
    return (alpha / np.pi) ** 0.25 * (-alpha * x) * xp.exp(-alpha * x ** 2 / 2)


def sderiv_angle(cds, hoh):
    # hoh = bond_angle(cds)
    g = gmat()
    alpha = theta_freq / g
    x = hoh - theta_eq
    return (alpha / np.pi) ** 0.25 * (alpha ** 2 * x ** 2 - alpha) * xp.exp(-alpha * x ** 2 / 2)


def trial_wavefunction(cds, ret_pdt=True):
    cds = xp.asarray(cds)
    psi = xp.zeros((3, len(cds)))
    ohs = oh_dists(cds)
    hoh = bond_angle(cds)
    for i in range(2):
        psi[i] = xp.interp(ohs[i], gridpts, wfn)
    psi[2] = angle(cds, hoh)
    if ret_pdt:
        prod_psi = xp.prod(psi, axis=0)
        return xp.asnumpy(prod_psi)
    else:
        # for use in chain rule. keep as cupy array
        return psi, ohs, hoh


def first_deriv(cds, ohs, hoh):
    """computes dpsi/dr1, dpsi/dr2, dpsi/dtheta"""
    dpsi = xp.zeros((3, len(cds)))
    for i in range(2):
        dpsi[i] = xp.interp(ohs[i], gridpts, der_wfn)
    dpsi[2] = deriv_angle(cds, hoh)
    return dpsi


def sec_deriv(cds, ohs, hoh):
    """computes dpsi/dr1, dpsi/dr2, dpsi/dtheta"""
    sdpsi = xp.zeros((3, len(cds)))
    ohs = oh_dists(cds)
    for i in range(2):
        sdpsi[i] = xp.interp(ohs[i], gridpts, sder_wfn)
    sdpsi[2] = sderiv_angle(cds, hoh)
    return sdpsi


def dpsi_dx(cds):
    """Retruns the first and second derivative"""
    import cupy as xp
    # First, calculate trial wave function (num_walkers x num_modes)
    cds = xp.asarray(cds)
    trl, ohs, hoh = trial_wavefunction(cds, ret_pdt=False)
    dpsi_dr = first_deriv(cds, ohs, hoh) / trl
    d2psi_dr2 = sec_deriv(cds, ohs, hoh) / trl

    ohs_idx = [[0, 2], [2, 1]]
    hoh_idx = [0, 2, 1]

    # Chain rule derivatives
    crh = pv.ChainRuleHelper(cds, xp)

    dr_dxs = xp.stack([crh.dr_dx(oh) for oh in ohs_idx])
    d2r_dx2s = xp.stack([crh.d2r_dx2(oh, dr_dx=dr_dxs[num]) for num, oh in enumerate(ohs_idx)])

    dcth_dx = crh.dcth_dx(hoh_idx,
                          dr_da=dr_dxs[0],
                          dr_dc=dr_dxs[1])
    dth_dx = crh.dth_dx(hoh_idx,
                        dcth_dx=dcth_dx,
                        dr_da=dr_dxs[0],
                        dr_dc=dr_dxs[1])

    d2th_dx2 = crh.d2th_dx2(hoh_idx,
                            dcth_dx=dcth_dx,
                            dr_da=dr_dxs[0],
                            dr_dc=dr_dxs[1],
                            d2r_da2=d2r_dx2s[0],
                            d2r_dc2=d2r_dx2s[1])
    # Calculate dpsi/dx / psi
    dint_dx = xp.concatenate((dr_dxs, xp.expand_dims(dth_dx, 0)))
    dp_dx = crh.dpsidx(dpsi_dr, dint_dx)

    # Calculate d2psi/dx2 / psi
    d2int_dx2 = xp.concatenate((d2r_dx2s, xp.expand_dims(d2th_dx2, 0)))
    d2p_dx2 = crh.d2psidx2(d2psi_dr2, d2int_dx2, dpsi_dr, dint_dx)
    return xp.asnumpy(dp_dx), xp.asnumpy(d2p_dx2)
