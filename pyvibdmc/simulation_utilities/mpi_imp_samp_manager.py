import numpy as np
import os, sys
import importlib
import itertools as itt
from itertools import repeat

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

from .mpi_potential_manager import MPI_Potential
from .initial_conditioner.finite_difference import MolFiniteDifference as MolFD
from .imp_samp import *

inited = False


class MPI_ImpSampManager:
    """Imports and Wraps around the user-provided trial wfn and (optionally) the first and second derivatives.
    Parallelized using MPI, uses MPI_Potential's infrastructure a little bit."""

    def __init__(self,
                 trial_function,
                 trial_directory,
                 python_file,
                 pot_manager,
                 deriv_function=None,
                 s_deriv_function=None,
                 trial_kwargs=None,
                 deriv_kwargs=None,
                 s_deriv_kwargs=None):

        self.trial_fuc = trial_function
        self.trial_dir = trial_directory
        self.python_file = python_file
        self.deriv_func = deriv_function
        self.sderiv_func = s_deriv_function
        self.trial_kwargs = trial_kwargs
        self.deriv_kwargs = deriv_kwargs
        self.sderiv_kwargs = s_deriv_kwargs
        self.pot_manager = pot_manager
        if not isinstance(self.pot_manager, MPI_Potential):
            raise ValueError("You can only use MPI imp sampling with an MPI potential. Sorry.")
        self.prep_imp(chdir=True)  # for main process

    def prep_imp(self, chdir=False):
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
            self.finite_diff = ImpSamp.finite_diff
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

    def initwrapper(self, cds, fun, kwargz=None):
        global inited
        if not inited:
            self.prep_imp(chdir=False)  # For child processes that are already in the directory
            inited = True
        if inited:
            if kwargz is None:
                rez = fun(cds)
            else:
                rez = fun(cds,kwargz)
        return rez

    def call_trial(self, cds):
        """Get trial wave function using MPI"""
        split_cds = np.array_split(cds, self.pot_manager.num_mpi)
        with MPICommExecutor() as executor:
            trialz = list(executor.map(self.initwrapper,
                                       split_cds, repeat(self.trial_wfn, len(split_cds)),
                                       repeat(self.trial_kwargs, len(split_cds))))
            trialz = np.concatenate(trialz)
        return trialz

    def call_trial_no_mp(self, cds):
        """For call_derivs, get trial wave function. Still used in the mp.pool context, just doesn't call pool itself"""
        if self.trial_kwargs is None:
            trial = self.trial_wfn(cds)
        else:
            trial = self.trial_wfn(cds, self.trial_kwargs)
        return trial

    def call_deriv(self, cds):
        """Call first derivative using MPI"""
        split_cds = np.array_split(cds, self.pot_manager.num_mpi)
        with MPICommExecutor() as executor:
            derivz = list(executor.map(self.initwrapper,
                                       split_cds, repeat(self.deriv, len(split_cds)),
                                       repeat(self.deriv_kwargs, len(split_cds))))

            derivz = np.concatenate(derivz)
        return derivz

    def call_sderiv(self, cds):
        """Call second derivative using MPI"""
        split_cds = np.array_split(cds, self.pot_manager.num_mpi)
        with MPICommExecutor() as executor:
            sderivz = list(executor.map(self.initwrapper,
                                        split_cds, repeat(self.sderiv, len(split_cds)),
                                        repeat(self.sderiv_kwargs, len(split_cds))))
            sderivz = np.concatenate(sderivz)
        return sderivz

    def call_derivs(self, cds):
        """For when derivatives are not supplied, call finite difference function.
        This is parallelized using MPI."""
        split_cds = np.array_split(cds, self.pot_manager.num_mpi)
        # derivz, sderivz = zip(*self.pool.starmap(self.finite_diff, zip(cds, repeat(self.call_trial_no_mp, len(cds)))))
        with MPICommExecutor() as executor:
            derivz,sderivz = zip(*list(executor.map(self.initwrapper,
                                                     split_cds,
                                                     repeat(self.finite_diff, len(split_cds)),
                                                     repeat(self.call_trial_no_mp, len(split_cds))
                                                     )
                                        )
                                  )

            derivz = np.concatenate(derivz)
            sderivz = np.concatenate(sderivz)
            # These if statements are for someone who supplied only first derv
            # function or only 2nd derv fuction but not both
            if self.sderiv is not None:
                cds = np.concatenate(cds)
                sderivz = self.call_sderiv(cds)
            if self.deriv is not None:
                cds = np.concatenate(cds)
                derivz = self.call_deriv(cds)
        return derivz, sderivz
