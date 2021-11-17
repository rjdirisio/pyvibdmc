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
        self.potential_function = potential_function
        self.python_file = python_file
        self.potential_directory = potential_directory
        self.pot_kwargs = pot_kwargs
        self._initialize()

    def _initialize(self):
        comm = MPI.COMM_WORLD
        self.num_mpi = comm.Get_size()
        print(f'MPI RANKS: {self.num_mpi}')

    def prep_pot(self):
        """Pretty much exactly what happens when mp pool is initialized in the potential_manager"""
        _curdir = os.getcwd()
        os.chdir(self.potential_directory)
        sys.path.insert(0, os.getcwd())
        module = self.python_file.split(".")[0]
        x = importlib.import_module(module)
        pot = getattr(x, self.potential_function)
        return pot

    def initwrapper(self, cds):
        """
        pre is prep_pot, which takes the tuple initargs that has the python file stuff in it.
        call_the_pot is the variable name for the  callpot function.
        cdz are the coordinates.
        """
        if len(cds.shape) == 2:
            cds = np.expand_dims(cds, 0)
        global inited, pt
        if not inited:
            pt = self.prep_pot()
            inited = True
        if self.pot_kwargs is None:
            vs = pt(cds)
        else:
            vs = pt(cds, self.pot_kwargs)
        return vs

    def getpot(self, cds, timeit=False):
        if timeit:
            start = time.time()
        with MPICommExecutor() as executor:
            result =  executor.map(self.initwrapper,
                                   cds,
                                   chunksize=self.num_mpi)
            v = np.array(list(result))
            v = np.squeeze(v)
        if timeit:
            elapsed = time.time() - start
            return v, elapsed
        else:
            return v
