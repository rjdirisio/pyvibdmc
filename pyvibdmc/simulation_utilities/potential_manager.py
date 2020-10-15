import multiprocessing as mp
import os
import sys
import time

import numpy as np


class Potential:
    """
    A potential handler that is able to call python functions that
    call .so files, either generated by f2py or loaded in by ctypes.
    :param potential_function: The name of a python function (user specified) that will take in a n x m x 3 stack of geometries and return a 1D numpy array filled with potential values in hartrees.
    :type potential_function: str
    :param potential_dir: The *absolute path* to the directory that contains the .so file and .py file. If it"s a python function, then just the absolute path to your .py file.
    :type: str
    :param num_cores: Will create a pool of <num_cores> processes using Python"s multiprocessing module. This should never be larger than the number of processors on the machine this code is run.
    :type: int
    """

    def __init__(self,
                 potential_function,
                 potential_directory,
                 python_file,
                 num_cores=1
                 ):
        self.pot_func = potential_function
        self.pyFile = python_file
        self.pot_dir = potential_directory
        self.num_cores = num_cores
        self._potPool = None
        self.init_pool()

    def _init_pot(self):
        """
        Sets _pot
        """
        import importlib
        # Go to potential directory that houses python function and assign a self._pot variable to it
        self._curdir = os.getcwd()
        os.chdir(self.pot_dir)
        sys.path.insert(0, os.getcwd())
        module = self.pyFile.split(".")[0]
        x = importlib.import_module(module)
        self._pot = getattr(x, self.pot_func)
        os.chdir(self._curdir)

    def pool_initalizer(self):
        self._init_pot()

    def init_pool(self):
        if self.num_cores > 1:
            self._potPool = mp.Pool(self.num_cores, initializer=self._init_pot())
        else:
            self._init_pot()

    def getpot(self, cds, timeit=False):
        """
        Uses the potential function we got to call potential
        :param cds: A stack of geometries (nxmx3, n=num_geoms;m=num_atoms;3=x,y,z) whose energies we need
        :type cds: np.ndarray
        :param timeit: The logger telling the potential manager whether or not to time the potential call
        :type timeit: bool
        """
        if timeit:
            start = time.time()
        if self._potPool is not None:
            cds = np.array_split(cds, self.num_cores)
            res = self._potPool.map(self._pot, cds)
            v = np.concatenate(res)
        else:
            v = self._pot(cds)
        if timeit:
            elapsed = time.time()-start
            return v, elapsed
        else:
            return v

    def mp_close(self):
        if self._potPool is not None:
            self._potPool.close()
            self._potPool.join()