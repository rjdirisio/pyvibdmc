import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import importlib
import sys, os
import time

inited = False
__all__ = ['MPI_Potential']


class MPI_Potential:
    def __init__(self,
                 potential_function,
                 potential_directory,
                 python_file,
                 pot_kwargs=None,
                 ):
        self.pot_func = potential_function
        self.pyFile = python_file
        self.pot_dir = potential_directory
        self.pot_kwargs = pot_kwargs
        self._initialize()

    def _initialize(self):
        comm = MPI.COMM_WORLD
        self.num_mpi = comm.Get_size()
        print(self.num_mpi)

    @staticmethod
    def prep_pot(pot_func, pyFile, pot_dir):
        """Pretty much exactly what happens when mp pool is initialized in the potential_manager"""
        _curdir = os.getcwd()
        os.chdir(pot_dir)
        sys.path.insert(0, os.getcwd())
        module = pyFile.split(".")[0]
        x = importlib.import_module(module)
        pot = getattr(x, pot_func)
        return pot

    @staticmethod
    def callpot(cds, pot):
        v = pot(cds)
        return v

    @staticmethod
    def initwrapper(pre, initargs, call_the_pot, cdz):
        """
        pre is prep_pot, which takes the tuple initargs that has the python file stuff in it.
        call_the_pot is the variable name for the  callpot function.
        cdz are the coordinates.
        """
        global inited, poot
        if not inited:
            poot = pre(*initargs)
            inited = True
        vs = call_the_pot(cdz, poot)
        return vs

    def getpot(self, cds, timeit=False):
        split_cds = np.array_split(cds, self.num_mpi)
        if timeit:
            start = time.time()
        with MPICommExecutor() as executor:
            result = list(executor.map(self.initwrapper,
                                       [self.prep_pot] * self.num_mpi,
                                       [(self.pot_func, self.pyFile, self.pot_dir)] * self.num_mpi,
                                       [self.callpot] * self.num_mpi,
                                       split_cds))
            v = np.concatenate(result)
        if timeit:
            elapsed = time.time() - start
            return v, elapsed
        else:
            return v