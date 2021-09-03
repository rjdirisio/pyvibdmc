import numpy as np
import os, sys
import importlib
from .potential_manager import Potential, Potential_NoMP, NN_Potential
from .initial_conditioner.finite_difference import MolFiniteDifference as MolFD
import itertools as itt
from itertools import repeat


class ImpSampManager:
    """Takes in a function that calls the trial wave function"""

    def __init__(self,
                 trial_function,
                 trial_directory,
                 python_file,
                 pot_manager,
                 deriv_function=None,
                 s_deriv_function=None,
                 trial_kwargs=None,
                 deriv_kwargs=None,
                 s_deriv_kwargs=None,
                 use_mpi=False):

        self.trial_fuc = trial_function
        self.trial_dir = trial_directory
        self.python_file = python_file
        self.deriv_func = deriv_function
        self.sderiv_func = s_deriv_function
        self.trial_kwargs = trial_kwargs
        self.deriv_kwargs = deriv_kwargs
        self.sderiv_kwargs = s_deriv_kwargs
        self.pot_manager = pot_manager
        self.use_mpi = use_mpi
        if isinstance(self.pot_manager, Potential):
            self.pool = self.pot_manager.pool
            # k = id(self.pool)
            # l = id(self.pot_manager.pool)
            self.num_cores = self.pot_manager.num_cores
            self._reinit_pool()
            # Thinking about assignment self.trial_wfn, self.deriv, ... to either be the MP, noMP or MPI versions
            # Which will allow me to then choose which one on the fly to use.
        elif isinstance(self.pot_manager, Potential_NoMP):
            pass
        else:
            try:
                from mpi_potential_manager import MPI_Potential
                print('setting up MPI for manager')
            except SyntaxError:
                print("Pass in a proper potential manager, dude")
            pass

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['pot_manager']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _init_wfn_mp(self, chdir=False):
        """Import the python functions of the pool workers on the pool.
        For when you have a Potential object"""
        if chdir:
            # For main process
            cur_dir = os.getcwd()
            os.chdir(self.trial_dir)
        sys.path.insert(0, os.getcwd())
        module = self.python_file.split(".")[0]
        x = importlib.import_module(module)
        self.trial_wfn = getattr(x, self.trial_fuc)
        if self.deriv_func is None or self.sderiv_func is None:
            # even if you have 1 or the other, do finite difference for everything,
            # bool for pyvibdmc sim code to do both derivs at once.
            self.all_finite = True
            self.derivs = self.finite_diff
            if self.deriv_func is not None:
                self.deriv = getattr(x, self.deriv_func)
            else:
                self.deriv = None
            if self.sderiv_func is not None:
                self.sderiv = getattr(x, self.sderiv_func)
            else:
                self.sderiv = None
        else:  # Supplied derivatives, just import them
            self.all_finite = False
            self.deriv = getattr(x, self.deriv_func)
            self.sderiv = getattr(x, self.sderiv_func)
        if chdir:
            # For main process
            os.chdir(cur_dir)

    def _reinit_pool(self):
        empt = [() for _ in range(self.num_cores)]
        self._init_wfn_mp(chdir=True)
        self.pot_manager.pool.starmap(self._init_wfn_mp, empt, chunksize=1)

    def call_mp_fun(self, cds, fun, kwargz):
        cds = np.array_split(cds, self.pot_manager.num_cores)
        if kwargz is not None:
            res = self.pool.starmap(fun, zip(cds, repeat(kwargz, len(cds))))
        else:
            res = self.pool.map(fun, cds)

        res = np.concatenate(res)
        return res

    def call_trial(self, cds):
        """Get trial wave function using multiprocessing"""
        trialz = self.call_mp_fun(cds, self.trial_wfn, self.trial_kwargs)
        return trialz

    def call_trial_no_mp(self, cds):
        """For call_derivs, get trial wave function. Still used in the mp.pool context, just doesn't call pool itself"""
        if self.trial_kwargs is None:
            trial = self.trial_wfn(cds)
        else:
            trial = self.trial_wfn(cds, self.trial_kwargs)
        return trial

    def call_deriv(self, cds):
        """Call first derivative using multiprocessing"""
        derivz = self.call_mp_fun(cds, self.deriv, self.deriv_kwargs)
        return derivz

    def call_sderiv(self, cds):
        """Call second derivative using multiprocessing"""
        sderivz = self.call_mp_fun(cds, self.sderiv, self.sderiv_kwargs)
        return sderivz

    def call_derivs(self, cds):
        """For when derivatives are not supplied, call finite difference function.
        This is still parallelized, but at the top level rather than at the trial wavefunction level."""
        cds = np.array_split(cds, self.num_cores)
        derivz, sderivz = zip(*self.pool.starmap(self.finite_diff, zip(cds, repeat(self.call_trial_no_mp, len(cds)))))
        return np.concatenate(derivz), np.concatenate(sderivz)

    @staticmethod
    def finite_diff(cds, trial_func):
        """Internal finite diff of needed derivatives for imp samp DMC, dpsi/dx and d2spi/dx2"""
        dx = 0.001  # Bohr
        num_atoms = cds.shape[1]
        first = np.zeros(cds.shape)
        sec = np.zeros(cds.shape)
        tmp_psi = np.zeros((len(cds), 3))
        tmp_psi[:, 1] = trial_func(cds)
        for atom_label in range(num_atoms):
            for xyz in range(3):
                cds[:, atom_label, xyz] -= dx
                tmp_psi[:, 0] = trial_func(cds)
                cds[:, atom_label, xyz] += 2. * dx
                tmp_psi[:, 2] = trial_func(cds)
                cds[:, atom_label, xyz] -= dx
                # Proper first derivative, which will be divided by 2 later
                first[:, atom_label, xyz] = (tmp_psi[:, 2] - tmp_psi[:, 0]) / (2 * dx)
                # Proper second derivative
                sec[:, atom_label, xyz] = ((tmp_psi[:, 0] - 2. * tmp_psi[:, 1] + tmp_psi[:, 2]) / dx ** 2)
        return first, sec


class ImpSamp:
    """Internal class that:
       1. Calculates local energy
       2. Calculates drift terms
    """

    def __init__(self, imp_samp_manager, finite_difference=False):
        self.imp_manager = imp_samp_manager
        self.findiff = finite_difference

    def trial(self, cds):
        trial_wfn = self.imp_manager.call_trial(cds)
        return trial_wfn

    def drift(self, cds):
        """Internally returns 2*(dpsi/psi), since it's more convenient in the workflow"""
        # num_walkers, num_atoms, 3 array
        psi_t = self.trial(cds)
        if self.findiff:
            deriv, sderiv = self.imp_manager.call_derivs(cds)
        else:
            deriv = self.imp_manager.call_deriv(cds)
            sderiv = None
        drift_term = deriv / psi_t[:, np.newaxis, np.newaxis]
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
        # inv_masses_trip is num_atoms, 3
        if sec_deriv is None:
            # num_walkers, num_atoms, 3 array
            sec_deriv = self.imp_manager.call_sderiv(cds)
        kinetic = -0.5 * np.sum(np.sum(inv_masses_trip * sec_deriv / psi_t[:, np.newaxis, np.newaxis], axis=1), axis=1)
        return kinetic
