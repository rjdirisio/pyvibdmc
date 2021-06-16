import h5py
import pickle
import copy
import glob

__all__ = ['SimArchivist']


class SimArchivist:
    """A utility class for saving wave functions, checkpoint files, and reloading DMC sims"""

    @staticmethod
    def save_h5(fname, keyz, valz):
        """
        Helper function to take in keys and values and save them in an hdf5 file
        :param fname: The name of the hdf5 file to be saved
        :param keyz: The keys used in saving the hdf5 file
        :param valz: The values that correspond to each key
        """
        with h5py.File(fname, 'w') as hf:
            for key, val in zip(keyz, valz):
                hf.create_dataset(key,
                                  data=val)

    @staticmethod
    def chkpt(dmc_obj, prop_step):
        """
        Given a DMC object and its current time step , this will save it as a .pickle file (excluding the potential!)
        """
        cheq = copy.deepcopy(dmc_obj)  # calls __deepcopy__
        with open(f'{dmc_obj.output_folder}/chkpts/{dmc_obj.sim_name}_{str(prop_step)}.pickle', 'wb') as handle:
            pickle.dump(cheq, handle, protocol=4)  # explicit protocol 4 to account for 3.8's upgrade to 5.

    @staticmethod
    def reload_sim(chkpt_folder, sim_name):
        """
        Given a .pickle file, reinitialize the DMC object and reassign potential.
        :return: DMC Object for one to run.
        """
        pickle_file = glob.glob(f"{chkpt_folder}/chkpts/{sim_name}_*.pickle")[0]
        with open(pickle_file, "rb") as handle:
            dmc_obj = pickle.load(handle)
        return dmc_obj
