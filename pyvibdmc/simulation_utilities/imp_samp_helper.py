import numpy as np
from ..analysis import AnalyzeWfn


class ChainRuleHelper:
    def __init__(self):
        self.dpsi_dr = None
        self.dr_dx = None

    @staticmethod
    def dpsidx(dpsi_dr, dr_dx):
        """Generic function that takes in a series of dpsi/dr matrices and dr/dx matrices and generates the dpsi/dx
         matrix. Assumes direct product wave function fed in appropriately."""
        dpsi_drp = dpsi_dr.transpose(1, 0)[:, np.newaxis, :, np.newaxis]
        dr_dxp = dr_dx.transpose(1, 2, 3, 0)
        jacob = np.matmul(dr_dxp, dpsi_drp).squeeze()
        return jacob

    @staticmethod
    def d2psidx2(d2psi_dr2, d2r_dx2, dpsi_dr, dr_dx):
        # jacob's "part 2"
        d2psi_dr2p = d2psi_dr2.transpose(1, 0)[:, np.newaxis, :, np.newaxis]
        dr_dxp = dr_dx.transpose(1, 2, 3, 0)
        term_1 = np.matmul(dr_dxp ** 2, d2psi_dr2p).squeeze()
        # jacob's "part 1"
        dpsi_drp = dpsi_dr.transpose(1, 0)[:, np.newaxis, :, np.newaxis]
        d2r_dx2p = d2r_dx2.transpose(1, 2, 3, 0)
        term_2 = np.matmul(d2r_dx2p, dpsi_drp).squeeze()
        # Finally, this last sum done in a numpy black magic way
        term_3 = 2 * np.matmul(dr_dxp * np.roll(dr_dxp, -1, axis=-1),
                           dpsi_drp * np.roll(dpsi_drp, -1, axis=2)).squeeze()
        return term_1 + term_2 + term_3

    @staticmethod
    def dr_dx(cds, atm_bonds):
        """
        calculates the dr/dx first derivatives for bond lengths
        :param cds: num_walkers x num_atms x 3 array
        :param atm_bonds: list of lists of bond lengths that are relevant to the trial wave function. like [[0,1],[0,2]]
        :return: len(atm_bonds) x num_walkers x num_atoms x 3 array of dr/dx
        """
        d_ar = np.zeros((len(atm_bonds),) + cds.shape)  # the first derivative array
        for num, atm_pair in enumerate(atm_bonds):
            diff_carts = cds[:, atm_pair[0]] - cds[:, atm_pair[1]]  # alpha values
            r = np.linalg.norm(diff_carts, axis=1)
            d_ar[num, :, atm_pair[0]] = diff_carts / r[:, np.newaxis]
            d_ar[num, :, atm_pair[1]] = -1 * diff_carts / r[:, np.newaxis]
        return d_ar

    @staticmethod
    def d2r_dx2(cds, atm_bonds, dr_dx=None):
        """
        Calculates d2r/dx2 second derivatives for bond lengths
        """
        d_ar = np.zeros((len(atm_bonds),) + cds.shape)  # the first derivative array
        for num, atm_pair in enumerate(atm_bonds):
            r = np.linalg.norm(cds[:, atm_pair[0]] - cds[:, atm_pair[1]], axis=1)[:, np.newaxis]
            if dr_dx is None:
                this_dr_dx = ChainRuleHelper.dr_dx(cds, [atm_pair])[0]
            else:
                this_dr_dx = dr_dx[num]
            d_ar[num, :, atm_pair[0]] = (1 / r) - (1 / r) * (this_dr_dx[:, atm_pair[0]]) ** 2
            d_ar[num, :, atm_pair[1]] = (1 / r) - (1 / r) * (this_dr_dx[:, atm_pair[1]]) ** 2
        return d_ar

    @staticmethod
    def dcth_dx(cds, atm_bonds):
        """
            calculates the dcos(theta)/dx first derivatives for bond lengths
            :param cds: num_walkers x num_atms x 3 array
            :param atm_bonds: list of lists of bond lengths that are relevant to the trial wave function. like [[0,2,1]].
            ALWAYS put vertex of angle in middle index atm_bonds[x][1].
            :param d_ar: Optional. The derivative array that can be built up starting with this one or in another function
            :return: len(atm_bonds) x num_walkers x num_atoms x 3 array of dr/dx
            """
        analyzer = AnalyzeWfn(cds)
        d_ar = np.zeros((len(atm_bonds),) + cds.shape)  # the first derivative array
        for num, atm_pair in enumerate(atm_bonds):
            cos_theta = np.cos(analyzer.bond_angle(atm_pair[0], atm_pair[1], atm_pair[2]))
            dr_da = ChainRuleHelper.dr_dx(cds, [[atm_pair[0], atm_pair[1]]])
            dr_dc = ChainRuleHelper.dr_dx(cds, [[atm_pair[1], atm_pair[2]]])

            # First do external atms, which are 0 and 2
            # need alpha_c - alpha_b
            alpha_1 = cds[:, atm_pair[2]] - cds[:, atm_pair[1]]
            # need rab and rcb
            rab = analyzer.bond_length(atm_pair[1], atm_pair[0])
            rcb = analyzer.bond_length(atm_pair[1], atm_pair[2])
            d_ar[num, :, atm_pair[0], :] = alpha_1 / (rab * rcb)[:, np.newaxis] - \
                                           (cos_theta / rab)[:, np.newaxis] * dr_da[0, :, atm_pair[0], :]
            # Next, need alpha_a and alpha_b
            alpha_2 = cds[:, atm_pair[0]] - cds[:, atm_pair[1]]
            d_ar[num, :, atm_pair[2], :] = alpha_2 / (rab * rcb)[:, np.newaxis] - \
                                           (cos_theta / rcb)[:, np.newaxis] * dr_dc[0, :, atm_pair[2], :]
            # Finally, do vertex atom
            alpha_b_a_c = (2 * cds[:, atm_pair[1]] - cds[:, atm_pair[0]] - cds[:, atm_pair[2]])
            d_ar[num, :, atm_pair[1], :] = alpha_b_a_c / (rab * rcb)[:, np.newaxis] - (cos_theta / rab)[:, np.newaxis] * \
                                           dr_da[0, :, atm_pair[1], :] - \
                                           (cos_theta / rcb)[:, np.newaxis] * dr_dc[0, :, atm_pair[1], :]
        return d_ar

    @staticmethod
    def d2cth_dx2(cds, atm_bonds):
        analyzer = AnalyzeWfn(cds)
        d_ar = np.zeros((len(atm_bonds),) + cds.shape)  # the first derivative array
        for num, atm_pair in enumerate(atm_bonds):
            # A bunch of things that are used throughout the calculation
            cos_theta = np.cos(analyzer.bond_angle(atm_pair[0], atm_pair[1], atm_pair[2]))
            dr_da = ChainRuleHelper.dr_dx(cds, [[atm_pair[0], atm_pair[1]]])
            dr_dc = ChainRuleHelper.dr_dx(cds, [[atm_pair[1], atm_pair[2]]])
            d2r_da2 = ChainRuleHelper.d2r_dx2(cds, [[atm_pair[0], atm_pair[1]]])
            d2r_dc2 = ChainRuleHelper.d2r_dx2(cds, [[atm_pair[1], atm_pair[2]]])
            rab = analyzer.bond_length(atm_pair[1], atm_pair[0])
            rcb = analyzer.bond_length(atm_pair[1], atm_pair[2])
            # First, get sec deriv wrt 1st ext atom
            alpha_1 = cds[:, atm_pair[2]] - cds[:, atm_pair[1]]
            term_1 = (-2 * alpha_1) / (rab ** 2 * rcb)[:, np.newaxis] * dr_da[0, :, atm_pair[0], :]
            term_2 = (2 * cos_theta / rab ** 2)[:, np.newaxis] * dr_da[0, :, atm_pair[0], :] ** 2
            term_3 = (-1 * cos_theta / rab)[:, np.newaxis] * d2r_da2[0, :, atm_pair[0], :]
            d_ar[num, :, atm_pair[0], :] = term_1 + term_2 + term_3
            # Do same thing for other external atom
            alpha_2 = cds[:, atm_pair[0]] - cds[:, atm_pair[1]]
            term_1 = (-2 * alpha_2) / (rab * rcb ** 2)[:, np.newaxis] * dr_dc[0, :, atm_pair[2], :]
            term_2 = (2 * cos_theta / rcb ** 2)[:, np.newaxis] * dr_dc[0, :, atm_pair[2], :] ** 2
            term_3 = (-1 * cos_theta / rcb)[:, np.newaxis] * d2r_dc2[0, :, atm_pair[2], :]
            d_ar[num, :, atm_pair[2], :] = term_1 + term_2 + term_3
            # Do the atm at vertex
            alpha_3 = 2 * cds[:, atm_pair[1]] - cds[:, atm_pair[0]] - cds[:, atm_pair[2]]
            term_1 = 2 / (rab * rcb)[:, np.newaxis]
            term_2 = (-1 * cos_theta / rab)[:, np.newaxis] * d2r_da2[0, :, atm_pair[1]]
            term_3 = (-1 * cos_theta / rcb)[:, np.newaxis] * d2r_dc2[0, :, atm_pair[1]]
            term_4 = (2 * cos_theta / rab ** 2)[:, np.newaxis] * dr_da[0, :, atm_pair[1]] ** 2
            term_5 = (2 * cos_theta / rcb ** 2)[:, np.newaxis] * dr_dc[0, :, atm_pair[1]] ** 2
            term_6 = (-2 * alpha_3 / (rab ** 2 * rcb)[:, np.newaxis]) * dr_da[0, :, atm_pair[1]]
            term_7 = (-2 * alpha_3 / (rab * rcb ** 2)[:, np.newaxis]) * dr_dc[0, :, atm_pair[1]]
            term_8 = (2 * cos_theta / (rab * rcb))[:, np.newaxis] * dr_da[0, :, atm_pair[1]] * dr_dc[0, :, atm_pair[1]]
            d_ar[num, :, atm_pair[1], :] = term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8
        return d_ar

    @staticmethod
    def dth_dx(cds, atm_bonds):
        analyzer = AnalyzeWfn(cds)
        dcos_ar = ChainRuleHelper.dcth_dx(cds, atm_bonds)
        d_ar = np.zeros((len(atm_bonds),) + cds.shape)  # the first derivative array
        for num, atm_pair in enumerate(atm_bonds):
            cos_theta = np.cos(analyzer.bond_angle(atm_pair[0], atm_pair[1], atm_pair[2]))
            dth_dcth = -1 / np.sqrt(1 - cos_theta ** 2)
            d_ar[num] = dth_dcth[:, np.newaxis, np.newaxis] * dcos_ar[num]
        return d_ar

    @staticmethod
    def d2th_dx2(cds, atm_bonds):
        analyzer = AnalyzeWfn(cds)
        dcos_ar = ChainRuleHelper.d2cth_dx2(cds, atm_bonds)
        d_ar = np.zeros((len(atm_bonds),) + cds.shape)  # the first derivative array
        for num, atm_pair in enumerate(atm_bonds):
            cos_theta = np.cos(analyzer.bond_angle(atm_pair[0], atm_pair[1], atm_pair[2]))
            # Calculate dth/dcth, and dcth/dx
            dth_dcth = -1 / np.sqrt(1 - cos_theta ** 2)
            dcth_dx = ChainRuleHelper.dcth_dx(cds, [atm_pair])[0]
            # Calculate d2th_dcos(th)
            d2th_dcth2 = -1 * cos_theta / ((1 - cos_theta ** 2) ** 1.5)
            term_1 = dcth_dx ** 2 * d2th_dcth2[:, np.newaxis, np.newaxis]
            term_2 = dcos_ar[num] * dth_dcth[:, np.newaxis, np.newaxis]
            # d_ar[num] = d2th_dcth2[:, np.newaxis, np.newaxis] * dcos_ar[num]
            d_ar[num] = term_1 + term_2
        return d_ar
