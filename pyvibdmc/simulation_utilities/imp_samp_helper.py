from ..analysis import AnalyzeWfn


class ChainRuleHelper:
    def __init__(self, coords, module):
        self.cds = coords
        self.xp = module
        self.analyzer = AnalyzeWfn(self.cds)

    def dpsidx(self, dpsi_dr, dr_dx):
        """Generic function that takes in a series of dpsi/dr matrices and dr/dx matrices and generates the dpsi/dx
         matrix. Assumes direct product wave function fed in appropriately."""
        dpsi_drp = dpsi_dr.transpose(1, 0)[:, self.xp.newaxis, :, self.xp.newaxis]
        dr_dxp = dr_dx.transpose(1, 2, 3, 0)
        jacob = self.xp.matmul(dr_dxp, dpsi_drp).squeeze()
        return jacob

    def d2psidx2(self,
                 d2psi_dr2,
                 d2r_dx2,
                 dpsi_dr,
                 dr_dx):
        """
        Generic function that takes in a series of d2psi_dr2 and d2r/dx matrices and generates the second derivative of
         psi wrt x, doing out the chain rule explicitly.
        """
        # jacob's "part 2"
        d2psi_dr2p = d2psi_dr2.transpose(1, 0)[:, self.xp.newaxis, :, self.xp.newaxis]
        dr_dxp = dr_dx.transpose(1, 2, 3, 0)
        term_1 = self.xp.matmul(dr_dxp ** 2, d2psi_dr2p).squeeze()
        # jacob's "part 1"
        dpsi_drp = dpsi_dr.transpose(1, 0)[:, self.xp.newaxis, :, self.xp.newaxis]
        d2r_dx2p = d2r_dx2.transpose(1, 2, 3, 0)
        term_2 = self.xp.matmul(d2r_dx2p, dpsi_drp).squeeze()
        # Finally, this last sum done in a numpy black magic way
        term_3 = 2 * self.xp.matmul(dr_dxp * self.xp.roll(dr_dxp, -1, axis=-1),
                                    dpsi_drp * self.xp.roll(dpsi_drp, -1, axis=2)).squeeze()
        return term_1 + term_2 + term_3

    def dr_dx(self, atm_pair):
        """
        calculates the dr/dx first derivatives for bond lengths
        :param self.cds: num_walkers x num_atms x 3 array
        :param atm_bonds: list of bond lengths that are relevant to the trial wave function. like [0,1]
        :return: num_walkers x num_atoms x 3 array of dr/dx
        """
        d_ar = self.xp.zeros(self.cds.shape)  # the first derivative array
        diff_carts = self.cds[:, atm_pair[0]] - self.cds[:, atm_pair[1]]  # alpha values
        r = self.xp.linalg.norm(diff_carts, axis=1)
        d_ar[:, atm_pair[0]] = diff_carts / r[:, self.xp.newaxis]
        d_ar[:, atm_pair[1]] = -1 * diff_carts / r[:, self.xp.newaxis]
        return d_ar

    def d2r_dx2(self,
                atm_pair,
                dr_dx=None):
        """
        Calculates d2r/dx2 second derivatives for bond lengths
        """
        d_ar = self.xp.zeros(self.cds.shape)  # the first derivative array
        r = self.xp.linalg.norm(self.cds[:, atm_pair[0]] - self.cds[:, atm_pair[1]], axis=1)[:, self.xp.newaxis]
        if dr_dx is None:
            dr_dx = self.dr_dx(atm_pair)
        d_ar[:, atm_pair[0]] = (1 / r) - (1 / r) * (dr_dx[:, atm_pair[0]]) ** 2
        d_ar[:, atm_pair[1]] = (1 / r) - (1 / r) * (dr_dx[:, atm_pair[1]]) ** 2
        return d_ar

    def dcth_dx(self,
                atm_pair,
                cos_theta=None,
                dr_da=None,
                dr_dc=None):
        """
            calculates the dcos(theta)/dx first derivatives for bond lengths
            :param self.cds: num_walkers x num_atms x 3 array
            :param atm_pair: list of bond lengths that are relevant to the trial wave function. like [0,2,1].
            ALWAYS put vertex of angle in middle index atm_bonds[x][1].
            :param dr_das: The derivative of the bond length corresponding to atm_bonds[i][0] and atm_bonds[i][1]. i is the index of the atm_bonds list of lists
            :param dr_dcs: The derivative of the bond length corresponding to atm_bonds[i][1] and atm_bonds[i][2]. i is the index of the atm_bonds list of lists
            :return: len(atm_bonds) x num_walkers x num_atoms x 3 array of dr/dx
            """

        d_ar = self.xp.zeros(self.cds.shape)  # the first derivative array
        if cos_theta is None:
            cos_theta = self.xp.cos(self.analyzer.bond_angle(atm_pair[0], atm_pair[1], atm_pair[2]))
        if dr_da is None or dr_dc is None:
            dr_da = self.dr_dx([atm_pair[0], atm_pair[1]])
            dr_dc = self.dr_dx([atm_pair[1], atm_pair[2]])
        # First do external atms, which are 0 and 2
        # need alpha_c - alpha_b
        alpha_1 = self.cds[:, atm_pair[2]] - self.cds[:, atm_pair[1]]
        # need rab and rcb
        rab = self.analyzer.bond_length(atm_pair[1], atm_pair[0])
        rcb = self.analyzer.bond_length(atm_pair[1], atm_pair[2])
        d_ar[:, atm_pair[0], :] = alpha_1 / (rab * rcb)[:, self.xp.newaxis] - \
                                  (cos_theta / rab)[:, self.xp.newaxis] * dr_da[:, atm_pair[0], :]
        # Next, need alpha_a and alpha_b
        alpha_2 = self.cds[:, atm_pair[0]] - self.cds[:, atm_pair[1]]
        d_ar[:, atm_pair[2], :] = alpha_2 / (rab * rcb)[:, self.xp.newaxis] - \
                                  (cos_theta / rcb)[:, self.xp.newaxis] * dr_dc[:, atm_pair[2], :]
        # Finally, do vertex atom
        alpha_b_a_c = (2 * self.cds[:, atm_pair[1]] - self.cds[:, atm_pair[0]] - self.cds[:, atm_pair[2]])
        d_ar[:, atm_pair[1], :] = alpha_b_a_c / (rab * rcb)[:, self.xp.newaxis] - (cos_theta / rab)[:,
                                                                                  self.xp.newaxis] * \
                                  dr_da[:, atm_pair[1], :] - \
                                  (cos_theta / rcb)[:, self.xp.newaxis] * dr_dc[:, atm_pair[1], :]
        return d_ar

    def d2cth_dx2(self,
                  atm_pair,
                  cos_theta=None,
                  dr_da=None,
                  dr_dc=None,
                  d2r_da2=None,
                  d2r_dc2=None):
        """
            :param self.cds: num_walkers x num_atms x 3 array
            :param atm_bonds: list of lists of bond lengths that are relevant to the trial wave function. like [[0,2,1]]. ALWAYS put vertex of angle in middle index atm_bonds[x][1].
            :param dr_das: The derivative of the bond length corresponding to atm_bonds[i][0] and atm_bonds[i][1]. i is the index of the atm_bonds list of lists
            :param dr_dcs: The derivative of the bond length corresponding to atm_bonds[i][1] and atm_bonds[i][2]. i is the index of the atm_bonds list of lists
            :param d2r_da2s: The second derivative of the bond length corresponding to atm_bonds[i][0] and atm_bonds[i][1]. i is the index of the atm_bonds list of lists
            :param d2r_dc2s: The second derivative of the bond length corresponding to atm_bonds[i][1] and atm_bonds[i][2]. i is the index of the atm_bonds list of lists
        """
        d_ar = self.xp.zeros(self.cds.shape)  # the first derivative array
        # A bunch of things that are used throughout the calculation
        if cos_theta is None:
            cos_theta = self.xp.cos(self.analyzer.bond_angle(atm_pair[0], atm_pair[1], atm_pair[2]))
        if dr_da is None or dr_dc is None:
            dr_da = self.dr_dx([atm_pair[0], atm_pair[1]])
            dr_dc = self.dr_dx([atm_pair[1], atm_pair[2]])
        if d2r_da2 is None or d2r_dc2 is None:
            d2r_da2 = self.d2r_dx2([atm_pair[0], atm_pair[1]])
            d2r_dc2 = self.d2r_dx2([atm_pair[1], atm_pair[2]])

        rab = self.analyzer.bond_length(atm_pair[1], atm_pair[0])
        rcb = self.analyzer.bond_length(atm_pair[1], atm_pair[2])
        # First, get sec deriv wrt 1st ext atom
        alpha_1 = self.cds[:, atm_pair[2]] - self.cds[:, atm_pair[1]]
        term_1 = (-2 * alpha_1) / (rab ** 2 * rcb)[:, self.xp.newaxis] * dr_da[:, atm_pair[0], :]
        term_2 = (2 * cos_theta / rab ** 2)[:, self.xp.newaxis] * dr_da[:, atm_pair[0], :] ** 2
        term_3 = (-1 * cos_theta / rab)[:, self.xp.newaxis] * d2r_da2[:, atm_pair[0], :]
        d_ar[:, atm_pair[0], :] = term_1 + term_2 + term_3
        # Do same thing for other external atom
        alpha_2 = self.cds[:, atm_pair[0]] - self.cds[:, atm_pair[1]]
        term_1 = (-2 * alpha_2) / (rab * rcb ** 2)[:, self.xp.newaxis] * dr_dc[:, atm_pair[2], :]
        term_2 = (2 * cos_theta / rcb ** 2)[:, self.xp.newaxis] * dr_dc[:, atm_pair[2], :] ** 2
        term_3 = (-1 * cos_theta / rcb)[:, self.xp.newaxis] * d2r_dc2[:, atm_pair[2], :]
        d_ar[:, atm_pair[2], :] = term_1 + term_2 + term_3
        # Do the atm at vertex
        alpha_3 = 2 * self.cds[:, atm_pair[1]] - self.cds[:, atm_pair[0]] - self.cds[:, atm_pair[2]]
        term_1 = 2 / (rab * rcb)[:, self.xp.newaxis]
        term_2 = (-1 * cos_theta / rab)[:, self.xp.newaxis] * d2r_da2[:, atm_pair[1]]
        term_3 = (-1 * cos_theta / rcb)[:, self.xp.newaxis] * d2r_dc2[:, atm_pair[1]]
        term_4 = (2 * cos_theta / rab ** 2)[:, self.xp.newaxis] * dr_da[:, atm_pair[1]] ** 2
        term_5 = (2 * cos_theta / rcb ** 2)[:, self.xp.newaxis] * dr_dc[:, atm_pair[1]] ** 2
        term_6 = (-2 * alpha_3 / (rab ** 2 * rcb)[:, self.xp.newaxis]) * dr_da[:, atm_pair[1]]
        term_7 = (-2 * alpha_3 / (rab * rcb ** 2)[:, self.xp.newaxis]) * dr_dc[:, atm_pair[1]]
        term_8 = (2 * cos_theta / (rab * rcb))[:, self.xp.newaxis] * dr_da[:, atm_pair[1]] * dr_dc[:, atm_pair[1]]
        d_ar[:, atm_pair[1], :] = term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8
        return d_ar

    def dth_dx(self,
               atm_pair,
               cos_theta=None,
               dcth_dx=None,
               dr_da=None,
               dr_dc=None):
        """
        :param self.cds: num_walkers x num_atoms x 3 array
        :param atm_pair: list that corresponds to indices of the ABC angle, where B is the vertex
        :param dcth_dxs: Derivative of cosine theta wrt x. This is needed for this derivative and the second derivative.
        """
        if dcth_dx is None:
            dcth_dx = self.dcth_dx(atm_pair, dr_da=dr_da, dr_dc=dr_dc)
        if cos_theta is None:
            cos_theta = self.xp.cos(self.analyzer.bond_angle(atm_pair[0], atm_pair[1], atm_pair[2]))
        dth_dcth = -1 / self.xp.sqrt(1 - cos_theta ** 2)
        d_ar = dth_dcth[:, self.xp.newaxis, self.xp.newaxis] * dcth_dx
        return d_ar

    def d2th_dx2(self,
                 atm_pair,
                 cos_theta=None,
                 dcth_dx=None,
                 dr_da=None,
                 dr_dc=None,
                 d2r_da2=None,
                 d2r_dc2=None):
        """
        :param self.cds: num_walkers x num_atoms x 3 array
        :param atm_pair: list that corresponds to indices of the ABC angle, where B is the vertex
        :param dcth_dxs: Derivative of cosine theta wrt x. This is needed for this derivative and the second derivative.
        """
        d2cth_dx2 = self.d2cth_dx2(atm_pair,
                                 dr_da=dr_da,
                                 dr_dc=dr_dc,
                                 d2r_da2=d2r_da2,
                                 d2r_dc2=d2r_dc2)
        if cos_theta is None:
            cos_theta = self.xp.cos(self.analyzer.bond_angle(atm_pair[0], atm_pair[1], atm_pair[2]))
        # Calculate dth/dcth, and dcth/dx
        dth_dcth = -1 / self.xp.sqrt(1 - cos_theta ** 2)
        if dcth_dx is None:
            dcth_dx = self.dcth_dx(atm_pair, dr_da=dr_da, dr_dc=dr_dc)
        # Calculate d2th_dcos(th)
        d2th_dcth2 = -1 * cos_theta / ((1 - cos_theta ** 2) ** 1.5)
        term_1 = dcth_dx ** 2 * d2th_dcth2[:, self.xp.newaxis, self.xp.newaxis]
        term_2 = d2cth_dx2 * dth_dcth[:, self.xp.newaxis, self.xp.newaxis]
        return term_1 + term_2
