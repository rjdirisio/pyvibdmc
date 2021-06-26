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
        print(f'MPI RANKS: {self.num_mpi}')

    def prep_pot(self):
        """Pretty much exactly what happens when mp pool is initialized in the potential_manager"""
        _curdir = os.getcwd()
        os.chdir(self.pot_dir)
        sys.path.insert(0, os.getcwd())
        module = self.pyFile.split(".")[0]
        x = importlib.import_module(module)
        pot = getattr(x, self.pot_func)
        return pot

    @staticmethod
    def callpot(cds, pot):
        v = pot(cds)
        return v

    def initwrapper(self, cdz):
        """
        pre is prep_pot, which takes the tuple initargs that has the python file stuff in it.
        call_the_pot is the variable name for the  callpot function.
        cdz are the coordinates.
        """
        global inited, poot
        if not inited:
            poot = self.prep_pot()
            inited = True
        vs = self.callpot(cdz, poot)
        return vs

    def getpot(self, cds, timeit=False):
        split_cds = np.array_split(cds, self.num_mpi)
        if timeit:
            start = time.time()
        with MPICommExecutor() as executor:
            result = list(executor.map(self.initwrapper,
                                       split_cds))
            v = np.concatenate(result)
        if timeit:
            elapsed = time.time() - start
            return v, elapsed
        else:
            return v