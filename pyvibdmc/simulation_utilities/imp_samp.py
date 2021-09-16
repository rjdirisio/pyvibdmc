from .imp_samp_manager import *
import numpy as np


class ImpSamp:
    """Internal class that:
       1. Calculates local energy
       2. Calculates drift terms
       3. Calls trial wave function
       Directly interfaces with the simulation. Also a good place to put finite difference derivatives.
    """

    def __init__(self, imp_samp_manager, finite_difference=False):
        self.imp_manager = imp_samp_manager
        self.findiff = finite_difference

    def trial(self, cds):
        """Internally returns the direct product wfn"""
        trial_wfn = self.imp_manager.call_trial(cds)
        if len(trial_wfn.shape) > 1:
            return np.prod(trial_wfn, axis=1)
        else:
            """Should only be for 1D problems"""
            return trial_wfn

    def drift(self, cds):
        """Internally returns 2*(dpsi/psi), since it's more convenient in the workflow"""
        # num_walkers, num_atoms, 3 array
        psi_t = self.trial(cds)
        if self.findiff:
            deriv, sderiv = self.imp_manager.call_derivs(cds)
            # Pre-divide by psi, as user is supposed to divide by psi if they provide derivatives.
            deriv = deriv / psi_t[:, np.newaxis, np.newaxis]
            sderiv = sderiv / psi_t[:, np.newaxis, np.newaxis]
        else:
            deriv = self.imp_manager.call_deriv(cds)
            sderiv = None
        # drift_term = deriv / psi_t[:, np.newaxis, np.newaxis]
        drift_term = deriv
        return 2 * drift_term, psi_t, sderiv

    @staticmethod
    def metropolis(sigma_trip,
                   trial_x,
                   trial_y,
                   disp_x,
                   disp_y,
                   f_x,
                   f_y):
        psi_ratio = (trial_y / trial_x) ** 2
        accep = np.exp(1. / 2. * (f_x + f_y) * (sigma_trip ** 2 / 4. * (f_x - f_y) - (disp_y - disp_x)))
        if accep.shape[-1] == 1:  # one-dimensional problem
            accep = accep.squeeze() * psi_ratio.squeeze()
        else:
            accep = np.prod(np.prod(accep, axis=1), axis=1) * psi_ratio
        return accep

    def local_kin(self, cds, inv_masses_trip, psi_t, sec_deriv=None):
        if sec_deriv is None:
            sec_deriv = self.imp_manager.call_sderiv(cds)
        # No division by psi as fin diff did it earlier and user provides sec derivs already divided by psi.
        # kinetic = -0.5 * np.sum(np.sum(inv_masses_trip * sec_deriv / psi_t[:, np.newaxis, np.newaxis], axis=1), axis=1)
        kinetic = -0.5 * np.sum(np.sum(inv_masses_trip * sec_deriv, axis=1), axis=1)
        return kinetic

    @staticmethod
    def get_tmp_psi(cds, trial_func, tmp_psi, tmp_psi_idx, prod):
        if prod:
            tmp_psi[:, tmp_psi_idx] = np.prod(trial_func(cds), axis=1)
        else:
            tmp_psi[:, tmp_psi_idx] = trial_func(cds)
        return tmp_psi

    @staticmethod
    def finite_diff(cds, trial_func):
        """Internal finite diff of needed derivatives for imp samp DMC, dpsi/dx and d2spi/dx2"""
        dx = 0.001  # Bohr. hard coded in baybee.
        num_atoms = cds.shape[1]
        num_dimz = cds.shape[-1]  # almost always 3
        first = np.zeros(cds.shape)
        sec = np.zeros(cds.shape)
        tmp_psi = np.zeros((len(cds), 3))
        tmp_trial = trial_func(cds)
        if len(tmp_trial.shape) > 1:
            prod = True
        else:
            prod = False
        tmp_psi = ImpSamp.get_tmp_psi(cds, trial_func, tmp_psi, 1, prod)
        for atom_label in range(num_atoms):
            for xyz in range(num_dimz):
                cds[:, atom_label, xyz] -= dx
                tmp_psi = ImpSamp.get_tmp_psi(cds, trial_func, tmp_psi, 0, prod)
                cds[:, atom_label, xyz] += 2. * dx
                tmp_psi = ImpSamp.get_tmp_psi(cds, trial_func, tmp_psi, 2, prod)
                cds[:, atom_label, xyz] -= dx
                # Proper first derivative, which will be divided by 2 later
                first[:, atom_label, xyz] = (tmp_psi[:, 2] - tmp_psi[:, 0]) / (2 * dx)
                # Proper second derivative
                sec[:, atom_label, xyz] = ((tmp_psi[:, 0] - 2. * tmp_psi[:, 1] + tmp_psi[:, 2]) / dx ** 2)
        return first, sec
