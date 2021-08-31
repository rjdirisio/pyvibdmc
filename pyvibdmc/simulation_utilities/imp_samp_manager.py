import numpy as np
import scipy as sp
import os, sys
from .potential_manager import Potential,Potential_NoMP,NN_Potential
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
            k = id(self.pool)
            l = id(self.pot_manager.pool)
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
            except ValueErorr:
                print("Pass in a proper potential manager, dude")
            pass

    def _init_wfn_mp(self):
        """Import the python functions of the pool workers on the pool.
        For when you have a Potential object"""
        sys.path.insert(0, os.getcwd())
        module = self.python_file.split(".")[0]
        x = importlib.import_module(module)
        self.trial_wfn = getattr(x, self.trial_fuc)
        if self.deriv_func is not None:
            self.deriv = getattr(x, self.deriv_func)
        if self.sderiv_func is not None:
            self.sderiv = getattr(x, self.sderiv_func)

    def _reinit_pool(self):
        empt = [() for _ in range(self.num_cores)]
        self.pool.starmap(self._init_wfn_mp, empt,chunksize=1)

    def call_mp_fun(self, cds, fun, kwargz):
        cds = np.array_split(cds, self.pot_manager.num_cores)
        if self.trial_kwargs is not None:
            res = self.pool.starmap(fun, zip(cds, repeat(kwargz, len(cds))))
        else:
            res = self.pool.map(fun, cds)
        trial = np.concatenate(res)
        return trial

    def call_trial(self, cds):
        trialz = self.call_mp_fun(cds, self.trial_wfn, self.trial_kwargs)
        return trialz

    def call_deriv(self, cds):
        derivz = self.call_mp_fun(cds, self.deriv, self.deriv_kwargs)
        return derivz

    def call_sderiv(self, cds):
        sderivz = self.call_mp_fun(cds, self.sderiv, self.sderiv_kwargs)
        return sderivz


class ImpSamp:
    """Internal class that:
       1. Calculates local energy
       2. Calculates drift terms
    """

    def __init__(self, imp_samp_manager):
        self.imp_manager = imp_samp_manager

    def trial(self, cds):
        trial_wfn = self.imp_manager.call_trial(cds)
        return trial_wfn

    def drift(self, cds):
        """Internally returns 2*(dpsi/psi), since it's more convenient in the workflow"""
        # num_walkers, num_atoms, 3 array
        psi_t = self.trial(cds)
        deriv = self.imp_manager.call_deriv(cds)
        drift_term = deriv / psi_t
        return 2 * drift_term

    @staticmethod
    def metropolis(self,
                   sigma_trip,
                   trial_x,
                   trial_y,
                   disp_x,
                   disp_y,
                   f_x,
                   f_y):
        psi_ratio = (trial_y / trial_x) ** 2
        accep = np.exp(1. / 2. * (f_x + f_y) * (sigma_trip ** 2 / 4. * (f_x - f_y) - (disp_y - disp_x)))
        accep = np.prod(np.prod(accep, axis=1), axis=1) * psi_ratio
        return accep

    def local_kin(self,cds,inv_masses_trip):
        #inv_masses_trip is num_atoms, 3
        #num_walkers, num_atoms, 3 array
        sec_deriv = self.imp_manager.call_sderiv(cds)
        # Check these axes.
        kinetic = -0.5 * np.sum(np.sum(inv_masses_trip[np.newaxis] * sec_deriv,axis=1),axis=1)
        pass
