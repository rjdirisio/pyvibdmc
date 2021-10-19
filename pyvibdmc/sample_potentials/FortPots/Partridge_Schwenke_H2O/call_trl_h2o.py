import pyvibdmc as pv
import numpy as np
from scipy import interpolate

"""Trial wave function wfn derivatives."""

r1_eq = pv.Constants.convert(0.95784, 'angstroms', to_AU=True)
r2_eq = pv.Constants.convert(0.95783997, 'angstroms', to_AU=True)
theta_eq = np.deg2rad(104.5080029)
theta_freq = pv.Constants.convert(1668.4590610594878, 'wavenumbers', to_AU=True)
inv_mh = 1 / pv.Constants.mass('H')
inv_mo = 1 / pv.Constants.mass('O')

# load in prepared, dense grid wave function
wvfn = np.load("free_oh_wvfn_dense.npy")
gridpts = wvfn[0]
wfn = wvfn[1]
der_wfn = wvfn[2]
sder_wfn = wvfn[3]

def dot_pdt(v1, v2):
    new_v1 = np.expand_dims(v1, axis=1)
    new_v2 = np.expand_dims(v2, axis=2)
    return np.matmul(new_v1, new_v2).squeeze()

def bond_angle(cds,ang_idx):
    vec1 = cds[:, ang_idx[0]] - cds[:, ang_idx[1]]
    vec2 = cds[:, ang_idx[2]] - cds[:, ang_idx[1]]
    dotV = np.arccos(dot_pdt(vec1, vec2) /
                     (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)))
    return dotV

def oh_dists(cds,idxs):
    oh1 = np.linalg.norm(cds[:, idxs[0]] - cds[:, idxs[1]], axis=1)
    return oh1


def gmat():
    return inv_mh / r1_eq ** 2 + inv_mh / r2_eq ** 2 + inv_mo * \
           (1 / r1_eq ** 2 + 1 / r2_eq ** 2 - 2 * np.cos(theta_eq) / (r1_eq * r2_eq))


def angles(hoh):
    g = gmat()
    alpha = theta_freq / g
    return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (hoh - theta_eq) ** 2 / 2)


def deriv_angles(hoh):
    g = gmat()
    alpha = theta_freq / g
    x = hoh - theta_eq
    return (alpha / np.pi) ** 0.25 * (-alpha * x) * np.exp(-alpha * x ** 2 / 2)


def sderiv_angles(hoh):
    g = gmat()
    alpha = theta_freq / g
    x = hoh - theta_eq
    return (alpha / np.pi) ** 0.25 * (alpha ** 2 * x ** 2 - alpha) * np.exp(-alpha * x ** 2 / 2)


def trial_wavefunction(cds, ex_args, ret_pdt=True):
    dists = ex_args['dists'] #list of lists
    angs = ex_args['angs'] #list of lists
    cds = np.asarray(cds)
    psi = np.zeros((len(dists) + len(angs), len(cds)))
    for i in range(len(dists)):
        ohs = oh_dists(cds, dists[i])
        psi[i] = np.interp(ohs,gridpts,wfn)
    for i in range(len(angs)):
        hoh = bond_angle(cds,angs[i])
        psi[i + len(dists)] = angles(hoh)
    if ret_pdt:
        prod_psi = np.prod(psi,axis=0)
        return prod_psi
    else:
        return psi


def first_deriv(ohs,hohs):
    """computes dpsi/dr1, dpsi/dr2, dpsi/dtheta"""
    dpsi = np.zeros((len(ohs) + len(hohs), len(ohs[0])))
    for i in range(len(ohs)):
        dpsi[i] = np.interp(ohs[i],gridpts,der_wfn)
    for i in range(len(hohs)):
        dpsi[i + len(ohs)] = deriv_angles(hohs[i])
    return dpsi


def sec_deriv(ohs,hohs):
    """computes d2psi/dr1, d2psi/dr2, d2psi/dtheta"""
    sdpsi = np.zeros((len(ohs) + len(hohs), len(ohs[0])))
    for i in range(len(ohs)):
        sdpsi[i] = np.interp(ohs[i],gridpts,sder_wfn)
    for i in range(len(hohs)):
        sdpsi[i + len(ohs)] = sderiv_angles(hohs[i])
    return sdpsi


def dpsi_dx(cds, ex_args):
    """Retruns the first and second derivative"""
    dists = ex_args['dists'] # list of lists
    nw = len(dists) // 2
    grouped_dists = np.reshape(np.asarray(dists),(nw,2,2)) # num_waters, num_ohs (always 2) , atms 0 and 1
    angs = ex_args['angs']

    ohz = np.asarray([oh_dists(cds,dist) for dist in dists])
    angz = np.asarray([bond_angle(cds,ang_idx=ang) for ang in angs])

    trl = trial_wavefunction(cds, ex_args, ret_pdt=False)

    dpsi_dr = first_deriv(ohz,angz) / trl
    d2psi_dr2 = sec_deriv(ohz,angz) / trl

    crh = pv.ChainRuleHelper(cds,np)

    dint_dx = np.zeros((len(angs) + len(dists),) + cds.shape)
    d2int_dx2 = np.zeros((len(angs) + len(dists),) + cds.shape)

    for num, hoh in enumerate(angs):
        ohs = grouped_dists[num]
        cos_theta = np.cos(angz[num]) 
        dr_dxs = np.stack([crh.dr_dx(oh) for oh in ohs])
        d2r_dx2s = np.stack([crh.d2r_dx2(oh, dr_dx=dr_dxs[num]) for num, oh in enumerate(ohs)])

        dcth_dx = crh.dcth_dx(hoh,
                              cos_theta=cos_theta,
                              dr_da=dr_dxs[0],
                              dr_dc=dr_dxs[1])
        dth_dx = crh.dth_dx(hoh,
                            cos_theta=cos_theta,
                            dcth_dx=dcth_dx,
                            dr_da=dr_dxs[0],
                            dr_dc=dr_dxs[1])
        d2th_dx2 = crh.d2th_dx2(hoh,
                                cos_theta=cos_theta,
                                dcth_dx=dcth_dx,
                                dr_da=dr_dxs[0],
                                dr_dc=dr_dxs[1],
                                d2r_da2=d2r_dx2s[0],
                                d2r_dc2=d2r_dx2s[1])
        dint_dx[num * 2: num * 2 + 2] = dr_dxs
        dint_dx[len(angs) * 2 + num] = dth_dx
        d2int_dx2[num * 2: num * 2 + 2] = d2r_dx2s
        d2int_dx2[len(angs) * 2 + num] = d2th_dx2
    dp_dx = crh.dpsidx(dpsi_dr, dint_dx)
    d2p_dx2 = crh.d2psidx2(d2psi_dr2, d2int_dx2, dpsi_dr, dint_dx)
    return dp_dx,d2p_dx2
