import matplotlib.pyplot as plt
from ..simulation_utilities import *


class SimInfo:
    """
    Object that takes in a .hdf5 file (one of the outputs of the simulation) and provides tools for analysis.

    :param h5Name: hdf5 file
    :type str
    """

    def __init__(self, h5Name):
        self.fname = h5Name
        self._initialize()
        self._load_sim_H5()

    def _initialize(self):
        wfn_name_temp = self.fname.split('/')
        sim_name = wfn_name_temp[-1]
        pth = '/'.join(wfn_name_temp[:-1])
        sim_name = sim_name.split('sim_info')[0]
        self._wfn_names = f"{pth}/wfns/{sim_name}wfn_"

    def _load_sim_H5(self):
        with h5py.File(self.fname, 'r') as f:
            self.vref_vs_tau = f['vref_vs_tau'][:]
            self.pop_vs_tau = f['pop_vs_tau'][:]

    @staticmethod
    def get_wfn(wfn_fl):
        """
        Given a. hdf5 file, return wave function and descendant weights associated with that wave function.

        :param wfn_fl: A resultant .hdf5 file from a PyVibDMC simulation
        :return: Coordinates array in angstroms (nxmx3), descendant weights array (n).
        """
        with h5py.File(wfn_fl, 'r') as f:
            cds = f['coords'][:]
            cds = Constants.convert(cds, 'angstroms', to_AU=False)
            dw = f['desc_weights'][:]
        return cds, dw

    def get_wfns(self, time_step_list):
        """
        Extract the wave function (walker set) and descendant weights given a time step number or numbers
        :param time_step_list: a list of ints that correspond to the time steps you want the wfn from given the simulation you are working with
        :type time_step_list: int or list
        :return:
        """
        time_step_list = [time_step_list] if isinstance(time_step_list, int) else time_step_list
        fl_list = [f'{self._wfn_names}{x}ts.hdf5' for x in time_step_list]
        tot_cds = []
        tot_dw = []
        for fl in fl_list:
            cds, dw = self.get_wfn(fl)
            tot_cds.append(cds)
            tot_dw.append(dw)
        tot_cds = np.concatenate(tot_cds)
        tot_dw = np.concatenate(tot_dw)
        return tot_cds, tot_dw

    def get_vref(self):
        return self.vref_vs_tau

    def get_pop(self):
        return self.pop_vs_tau

    def get_zpe(self, onwards=1000):
        """onwards is an int that tells us where to start averaging (python indexing
        starts at 0)"""
        return np.average(self.vref_vs_tau[onwards:, 1])

    def window_avg(self, blocks=5):
        """Splits vref into """
