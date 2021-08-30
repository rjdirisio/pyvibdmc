import numpy as np
import scipy as sp
import os,sys

class ImpSamp:
    """Takes in a function that calls the trial wave function"""

    def __init__(self,
                 trial_function,
                 trial_directory,
                 python_file,
                 deriv_function=None,
                 s_deriv_function=None,
                 pool=None,
                 use_mpi=False):

        self.trial_fuc = trial_function
        self.trial_dir = trial_directory
        self.python_file = python_file
        self.deriv_func = deriv_function
        self.sderiv_func = s_deriv_function
        self.pool = pool
        self.use_mpi=use_mpi
        if self.pool is not None:
            self._reinit_pool()

    def _init_wfn(self):
        """Import the python functions of the pool workers on the pool"""
        sys.path.insert(0, os.getcwd())
        module = self.python_file.split(".")[0]
        x = importlib.import_module(module)
        self.trial_wfn = getattr(x, self.trial_fuc)
        if self.deriv_func is not None:
            self.deriv = getattr(x, self.deriv_func)
        if self.sderiv_func is not None:
            self.sderiv = getattr(x, self.sderiv_func)

    def _reinit_pool(self):
        self.pool.map()

    def get_psi(self):
        pass


class GuidedDMCManager:
    """Internal class that:
       1. Calculates local energy
       2. Calculates drift terms
    """

    def __init__(self, t_wfn_manager):
        pass
