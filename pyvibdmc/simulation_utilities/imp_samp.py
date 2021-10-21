from .imp_samp_manager import *
import numpy as np


class ImpSamp:
    """Internal class that:
       1. Calculates local energy
       2. Calculates drift terms
       3. Calls trial wave function
       Directly interfaces with the simulation. Also a good place to put finite difference derivatives.
    """

    def __init__(self, imp_samp_manager):
        self.imp_manager = imp_samp_manager

    def trial(self, cds):
        """Internally returns the direct product wfn"""
        trial_wfn = self.imp_manager.call_trial(cds)
        return trial_wfn

    def drift(self, cds):
        """Internally returns (dpsi/psi), since it's more convenient in the workflow.
        Also returns second derivatives divided by psi"""
        psi_t = self.trial(cds)
        deriv, sderiv = self.imp_manager.call_derivs(cds)  # num_walkers, num_atoms, 3 array
        return deriv, psi_t, sderiv

    @staticmethod
    def metropolis(sigma_trip,
                   trial_x,
                   trial_y,
                   disp_x,
                   disp_y,
                   D_x,
                   D_y,
                   dt):
        psi_ratio = (trial_y / trial_x) ** 2
        term_1 = np.exp(-1 * (disp_x - disp_y - D_y * dt) ** 2 / (2 * sigma_trip ** 2))
        term_2 = np.exp(-1 * (disp_y - disp_x - D_x * dt) ** 2 / (2 * sigma_trip ** 2))
        accep = term_1 / term_2
        if accep.shape[-1] == 1:  # one-dimensional problem
            accep = accep.squeeze() * psi_ratio.squeeze()
        else:
            accep = np.prod(np.prod(accep, axis=1), axis=1) * psi_ratio
        flipped = np.where(trial_x * trial_y <= 0)[0]
        accep[flipped] = 0.0
        return accep

    @staticmethod
    def local_kin(inv_masses_trip, sec_deriv):
        # No division by psi as fin diff did it earlier and user provides sec derivs already divided by psi.
        kinetic = -0.5 * np.sum(np.sum(inv_masses_trip * sec_deriv, axis=1), axis=1)
        return kinetic

    @staticmethod
    def finite_diff(cds, trial_func):
        """Internal finite diff of needed derivatives for imp samp DMC, dpsi/dx and d2spi/dx2"""
        dx = 0.001  # Bohr. hard coded in baybee.
        num_atoms = cds.shape[1]
        num_dimz = cds.shape[-1]  # almost always 3
        first = np.zeros(cds.shape)
        sec = np.zeros(cds.shape)
        tmp_psi = np.zeros((len(cds), 3))
        tmp_psi[:, 1] = trial_func(cds)
        for atom_label in range(num_atoms):
            for xyz in range(num_dimz):
                cds[:, atom_label, xyz] -= dx
                tmp_psi[:, 0] = trial_func(cds)
                cds[:, atom_label, xyz] += 2. * dx
                tmp_psi[:, 2] = trial_func(cds)
                cds[:, atom_label, xyz] -= dx
                # Pure first derivative, which will be divided by 2 later
                first[:, atom_label, xyz] = (tmp_psi[:, 2] - tmp_psi[:, 0]) / (2 * dx)
                # Pure second derivative
                sec[:, atom_label, xyz] = ((tmp_psi[:, 0] - 2. * tmp_psi[:, 1] + tmp_psi[:, 2]) / dx ** 2)
        return first, sec, tmp_psi[:, 1]
