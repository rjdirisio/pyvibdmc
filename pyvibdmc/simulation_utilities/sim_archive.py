import h5py
import numpy as np
import pickle


class SimArchivist:
    """A utility class for saving wave functions, checkpoint files, and reloading DMC sims"""

    @staticmethod
    def saveH5(fname, keyz, valz):
        with h5py.File(fname, 'w') as hf:
            for key, val in zip(keyz, valz):
                hf.create_dataset(key,
                                  data=val)

    @staticmethod
    def chkpt(dmcObj, prop_step):
        with open(f'{dmcObj.output_folder}/chkpt_{dmcObj.sim_name}_{str(prop_step)}.pickle', 'wb') as handle:
            pickle.dump(dmcObj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def reloadSim(chkpt_folder, sim_name, time_step):
        with open(f'{chkpt_folder}/chkpt_{sim_name}_{str(time_step)}.pickle', 'rb') as handle:
            return pickle.load(handle)
