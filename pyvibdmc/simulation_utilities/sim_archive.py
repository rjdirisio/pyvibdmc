import h5py
import pickle
import copy


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
    def chkpt(dmcObj, prop_step):
        """
        Given a DMC object and its current time step , this will save it as a .pickle file (excluding the potential!)
        """
        cheq = copy.deepcopy(dmcObj)  # calls __deepcopy__
        with open(f'{dmcObj.output_folder}/chkpts/{dmcObj.sim_name}_{str(prop_step)}.pickle', 'wb') as handle:
            pickle.dump(cheq, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def reload_sim(chkpt_folder, sim_name, time_step):
        """
        Given a .pickle file, reinitialize the DMC object and reassign potential.
        :return: DMC Object for one to run.
        """
        with open(f"{chkpt_folder}/chkpts/{sim_name}_{time_step}.pickle", "rb") as handle:
            dmcObj = pickle.load(handle)
        return dmcObj
