import numpy as np
import os, sys
import importlib
import itertools as itt
from itertools import repeat

from .potential_manager import Potential, Potential_NoMP, NN_Potential
from .imp_samp import *


class ImpSampManager:
    """Imports and Wraps around the user-provided trial wfn and (optionally) the first and second derivatives.
    Parallelized using multiprocessing, which is considered the default for pyvibdmc."""

    def __init__(self,
                 trial_function,
                 trial_directory,
                 python_file,
                 pot_manager,
                 pass_timestep=False,
                 new_pool_num_cores=None,
                 deriv_function=None,
                 trial_kwargs=None,
                 deriv_kwargs=None):
        self.trial_func = trial_function
        self.trial_dir = trial_directory
        self.python_file = python_file
        self.deriv_func = deriv_function
        self.trial_kwargs = trial_kwargs
        self.deriv_kwargs = deriv_kwargs
        self.pot_manager = pot_manager
        self.pass_timestep = pass_timestep
        self.nomp_pool_cores = new_pool_num_cores  # Only when one wants to do multiprocessing importance sampling with noMP potential (like NN-DMC)
        if self.pass_timestep:
            self.ct = 0
            self.trial_kwargs['timestep']=0
            self.deriv_kwargs['timestep']=0

        if isinstance(self.pot_manager, Potential):
            self.pool = self.pot_manager.pool
            self.num_cores = self.pot_manager.num_cores
            self._reinit_pool()
        elif (isinstance(self.pot_manager, Potential_NoMP) or isinstance(self.pot_manager,
                                                                         NN_Potential)) and self.nomp_pool_cores is not None:
            """Really only for NN_Potential using multi-core imp samp"""
            from multiprocessing import Pool
            self.pool = Pool(self.nomp_pool_cores)
            self.num_cores = self.nomp_pool_cores
            self._reinit_pool()

    def __getstate__(self):
        """Since pool is a variable inside this class, the object cannot be pickled + used for multiprocessing.
        The solution is to use __getstate__/__setstate, which will delete the pool and pot_manager internally
         when needed."""
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['pot_manager']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _init_wfn_mp(self, chdir=False):
        """Import the python functions of the pool workers on the pool.
        For when you have a Potential object. For simplicity, efficiency, and restrictiveness, the imp samp stuff
        should be in the same directory as the potential energy callers."""
        if chdir:
            # For main process
            cur_dir = os.getcwd()
            os.chdir(self.trial_dir)
        sys.path.insert(0, os.getcwd())
        module = self.python_file.split(".")[0]
        x = importlib.import_module(module)
        self.trial_wfn = getattr(x, self.trial_func)
        if self.deriv_func is None:
            # bool for pyvibdmc sim code to do both derivs at once.
            self.all_finite = True
            self.derivs = ImpSamp.finite_diff
        else:  # Supplied derivatives, just import them
            self.all_finite = False
            self.derivs = getattr(x, self.deriv_func)
        if chdir:
            # For main process
            os.chdir(cur_dir)

    def _reinit_pool(self):
        """Imports the appropriate modules that are in the potential_manager directory"""
        empt = [() for _ in range(self.num_cores)]
        self._init_wfn_mp(chdir=True)
        self.pot_manager.pool.starmap(self._init_wfn_mp, empt, chunksize=1)

    def call_trial(self, cds):
        """Get trial wave function using multiprocessing"""
        cds = np.array_split(cds, self.pot_manager.num_cores)
        if self.trial_kwargs is not None:
            res = self.pool.starmap(self.trial_wfn, zip(cds, repeat(self.trial_kwargs, len(cds))))
        else:
            res = self.pool.map(self.trial_wfn, cds)
        res = np.concatenate(res)
        return res

    def call_trial_no_mp(self, cds):
        """For call_derivs (finite diff), get trial wave function.
         Still used in the mp.pool context, just doesn't call pool itself"""
        if self.trial_kwargs is None:
            trial = self.trial_wfn(cds)
        else:
            trial = self.trial_wfn(cds, self.trial_kwargs)
        return trial

    def call_derivs(self, cds):
        """For when derivatives are not supplied, call finite difference function.
        This is still parallelized."""
        cds = np.array_split(cds, self.num_cores)
        if self.all_finite:
            # Divide by trial wfn if finite difference
            derivz, sderivz, trial_wfn = zip(*self.pool.starmap(self.derivs,
                                                                zip(cds, repeat(self.call_trial_no_mp, len(cds)))))
            derivz = np.concatenate(derivz) / np.concatenate(trial_wfn)[:, np.newaxis, np.newaxis]
            sderivz = np.concatenate(sderivz) / np.concatenate(trial_wfn)[:, np.newaxis, np.newaxis]
        else:
            if self.deriv_kwargs is None:
                derivz, sderivz = zip(*self.pool.map(self.derivs, cds))
            else:
                derivz, sderivz = zip(*self.pool.starmap(self.derivs, zip(cds, repeat(self.deriv_kwargs, len(cds)))))
            derivz = np.concatenate(derivz)
            sderivz = np.concatenate(sderivz)
            ##Testing
            # fderivz, fsderivz, trial_wfn = ImpSamp.finite_diff(np.concatenate(cds), trial_func=self.call_trial_no_mp)
            # fderivz = fderivz / trial_wfn[:, np.newaxis, np.newaxis]
            # fsderivz = fsderivz / trial_wfn[:, np.newaxis, np.newaxis]
            # print('deriv:', np.average(fderivz-derivz))
            # print('sderiv:', np.average(fsderivz-sderivz))
            # print('hi')
        if self.pass_timestep:
            self.ct+=1
            self.trial_kwargs['timestep'] = self.ct
            self.deriv_kwargs['timestep'] = self.ct
        return derivz, sderivz


class ImpSampManager_NoMP:
    """Version of the manager that does not use any multiprocessing. If we ever evaluate the trial wfns with GPUs
    this could be useful. Could also be useful if multiprocessing is incompatible with your workflow."""

    def __init__(self,
                 trial_function,
                 trial_directory,
                 python_file,
                 chdir=False,
                 pass_timestep=False,
                 deriv_function=None,
                 trial_kwargs=None,
                 deriv_kwargs=None, ):
        self.trial_fuc = trial_function
        self.trial_dir = trial_directory
        self.python_file = python_file
        self.pass_timestep = pass_timestep
        self.deriv_func = deriv_function
        self.trial_kwargs = trial_kwargs
        self.deriv_kwargs = deriv_kwargs
        self.chdir = chdir
        if self.pass_timestep:
            self.ct = 0
            self.trial_kwargs['timestep']=0
            self.deriv_kwargs['timestep']=0

        self._import_modz()

    def _import_modz(self):
        self._curdir = os.getcwd()
        os.chdir(self.trial_dir)
        sys.path.insert(0, os.getcwd())
        module = self.python_file.split(".")[0]
        x = importlib.import_module(module)
        self.trial = getattr(x, self.trial_fuc)
        if self.deriv_func is None:
            self.all_finite = True
            self.derivs = ImpSamp.finite_diff
        else:  # Supplied derivatives, just import them
            self.all_finite = False
            self.derivs = getattr(x, self.deriv_func)
        os.chdir(self._curdir)

    def call_imp_func(self, func, cds, func_kwargs=None):
        """Convenience function for trial, deriv, and sderiv so I don't have to have triplicates of code"""
        if self.chdir:
            os.chdir(self.trial_dir)
        if func_kwargs is None:
            ret_val = func(cds)
        else:
            ret_val = func(cds, func_kwargs)
        if self.chdir:
            os.chdir(self._curdir)
        return ret_val

    def call_trial(self, cds):
        """Call trial wave function."""
        trial = self.call_imp_func(self.trial, cds, self.trial_kwargs)
        return trial

    def call_derivs(self, cds):
        """For when derivatives are not supplied, call finite difference function.
        Returns derivatives divided by psi already"""
        if self.all_finite:
            derivz, sderivz, trial_wfn = self.derivs(cds, trial_func=self.call_trial)
            derivz = derivz / trial_wfn[:, np.newaxis, np.newaxis]
            sderivz = sderivz / trial_wfn[:, np.newaxis, np.newaxis]
        else:
            derivz, sderivz = self.call_imp_func(self.derivs, cds, self.deriv_kwargs)
            ###Testing
            # fderivz, fsderivz, trial_wfn = ImpSamp.finite_diff(cds, trial_func=self.call_trial)
            # fderivz = fderivz / trial_wfn[:, np.newaxis, np.newaxis]
            # fsderivz = fsderivz / trial_wfn[:, np.newaxis, np.newaxis]
            # max_d = np.average(fderivz - derivz)
            # max_sd = np.average(fsderivz - sderivz)
            # print(f"Avg Psi: {max_d}")
            # print(f"Avg 2Psi: {max_sd}")
            ###/Testing
        if self.pass_timestep:
            self.ct+=1
            self.trial_kwargs['timestep'] = self.ct
            self.deriv_kwargs['timestep'] = self.ct
        return derivz, sderivz
