import h5py
import numpy as np

from ..simulation_utilities import *

__all__ = ['SimInfo']


class SimInfo:
    """
    Object that takes in a .hdf5 file (one of the outputs of the simulation) and provides tools for analysis.

    :param h5Name: hdf5 file
    :type str
    """

    def __init__(self, h5_name):
        self.fname = h5_name
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
            self.atom_nums = f['atomic_nums'][:]
            self.atom_masses = f['atomic_masses'][:]

    @staticmethod
    def get_wfn(wfn_fl, ret_ang=False, get_parent_wts=False):
        """
        Given a .hdf5 file, return wave function and descendant weights associated with that wave function.

        :param wfn_fl: A resultant .hdf5 file from a PyVibDMC simulation
        :param ret_ang: boolean indicating returning the coordinates in angtstroms. Bohr is the default.
        :return: Coordinates array in angstroms (nxmx3), descendant weights array (n).
        """
        with h5py.File(wfn_fl, 'r') as f:
            cds = f['coords'][:]
            if ret_ang:
                # Fenris said it was dumb to convert, let the user decide what to do
                cds = Constants.convert(cds, 'angstroms', to_AU=False)
            dw = f['desc_wts'][:]
            if get_parent_wts:
                parent_wts =  f['parent_wts'][:]
                return cds, dw, parent_wts
        return cds, dw

    def get_wfns(self, time_step_list, ret_ang=False, get_parent_wts=False):
        """
        Extract the wave function (walker set) and descendant weights given a time step number or numbers
        :param time_step_list: a list of ints that correspond to the time steps you want the wfn from given the simulation you are working with
        :type time_step_list: int or list
        :param ret_ang: boolean indicating returning the coordinates in angtstroms. Bohr is the default.
        :param get_parent_wts: Return the continuous weights associated with the walkers at the beginning of descendant weighting.
        """
        time_step_list = [time_step_list] if isinstance(time_step_list, int) else time_step_list
        fl_list = [f'{self._wfn_names}{x}ts.hdf5' for x in time_step_list]
        tot_cds = []
        tot_dw = []
        parent_wts = []
        for fl in fl_list:
            if get_parent_wts:
                cds, dw, par_wts = self.get_wfn(fl, ret_ang, get_parent_wts)
                parent_wts.append(par_wts)
            else:
                cds, dw = self.get_wfn(fl, ret_ang, get_parent_wts)
            tot_cds.append(cds)
            tot_dw.append(dw)
        tot_cds = np.concatenate(tot_cds)
        tot_dw = np.concatenate(tot_dw)
        if get_parent_wts:
            tot_parent_wts = np.concatenate(parent_wts)
            return tot_cds, tot_dw, tot_parent_wts
        else:
            return tot_cds, tot_dw

    def get_vref(self, ret_cm=False):
        """Returns vref_vs_tau array"""
        if ret_cm:
            vref_cm = Constants.convert(self.vref_vs_tau[:, 1], 'wavenumbers', to_AU=False)
            return np.column_stack((self.vref_vs_tau[:, 0], vref_cm))
        else:
            return self.vref_vs_tau

    def get_pop(self):
        """Returns population array, either ensemble size or sum of weights"""
        return self.pop_vs_tau

    def get_atomic_nums(self):
        """Returns list of atoms used in the simulation (by atomic number)"""
        return self.atom_nums

    def get_atom_masses(self):
        """Returns masses used in the simulation in atomic units (mass electron)"""
        return self.atom_masses

    def get_zpe(self, onwards=1000, ret_cm=False):
        """onwards is an int that tells us where to start averaging (python indexing
        starts at 0)"""
        if ret_cm:
            return Constants.convert(np.average(self.vref_vs_tau[onwards:, 1]), 'wavenumbers', to_AU=False)
        else:
            return np.average(self.vref_vs_tau[onwards:, 1])

    def window_avg(self, blocks=5, ret_cm=False):
        """Splits vref into blocks, calculates zpe in each of those blocks"""
        chunks = np.array_split(self.vref_vs_tau, blocks)
        avgs = np.average(chunks, axis=1)
        if ret_cm:
            return Constants.convert(avgs, 'wavenumbers', to_AU=False)
        else:
            return avgs

    @staticmethod
    def get_training(training_file, ret_ang=False, ret_cm=False):
        """If using deb_training_every argument, read the files with this. Returns walkers in angstr and engs in cm-1"""
        with h5py.File(training_file, 'r') as f:
            cds = f['coords'][:]
            # Fenris said it was dumb to convert, let the user decide what to do with Bohr and Hartree
            if ret_ang:
                cds = Constants.convert(cds, 'angstroms', to_AU=False)
            pots = f['pots'][:]
            if ret_cm:
                pots = Constants.convert(pots, 'wavenumbers', to_AU=False)
        return cds, pots
