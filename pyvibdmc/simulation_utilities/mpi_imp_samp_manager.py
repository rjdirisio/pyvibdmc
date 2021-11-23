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
                 pass_timestep=False,
                 deriv_function=None,
                 trial_kwargs=None,
                 deriv_kwargs=None):

        self.trial_fuc = trial_function
        self.trial_dir = trial_directory
        self.python_file = python_file
        self.pass_timestep = pass_timestep
        self.deriv_func = deriv_function
        self.trial_kwargs = trial_kwargs
        self.deriv_kwargs = deriv_kwargs
        self.pot_manager = pot_manager
        if not isinstance(self.pot_manager, MPI_Potential):
            raise ValueError("You can only use MPI imp sampling with an MPI potential. Sorry.")
        if self.pass_timestep:
            self.ct = 0
            self.trial_kwargs['timestep']=0
            self.deriv_kwargs['timestep']=0
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

    def initwrapper(self, cds, fun, kwargz=None):
        global inited
        if not inited:
            self.prep_imp(chdir=False)  # For child processes that are already in the directory
            inited = True
        if inited:
            if kwargz is None:
                rez = fun(cds)
            else:
                rez = fun(cds, kwargz)
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

    def call_derivs(self, cds):
        """For when derivatives are not supplied, call finite difference function.
        This is parallelized using MPI."""
        split_cds = np.array_split(cds, self.pot_manager.num_mpi)
        if self.all_finite:
            with MPICommExecutor() as executor:
                derivz, sderivz, trial_wfn = zip(*list(executor.map(self.initwrapper,
                                                                    split_cds,
                                                                    repeat(self.derivs, len(split_cds)),
                                                                    repeat(self.call_trial_no_mp, len(split_cds)))))
            trial_wfn = np.concatenate(trial_wfn)
            derivz = np.concatenate(derivz) / trial_wfn[:, np.newaxis, np.newaxis]
            sderivz = np.concatenate(sderivz) / trial_wfn[:, np.newaxis, np.newaxis]
        else:
            with MPICommExecutor() as executor:
                derivz, sderivz = zip(*list(executor.map(self.initwrapper,
                                                         split_cds, repeat(self.derivs, len(split_cds)),
                                                         repeat(self.deriv_kwargs, len(split_cds)))))
                derivz = np.concatenate(derivz)
                sderivz = np.concatenate(sderivz)
        if self.pass_timestep:
            self.ct+=1
            self.trial_kwargs['timestep'] = self.ct
            self.deriv_kwargs['timestep'] = self.ct
        return derivz, sderivz
