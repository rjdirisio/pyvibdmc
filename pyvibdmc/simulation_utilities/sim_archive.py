import h5py
import pickle
import copy

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
        cheq = copy.deepcopy(dmcObj)
        with open(f'{dmcObj.output_folder}/chkpts/{dmcObj.sim_name}_{str(prop_step)}.pickle', 'wb') as handle:
            pickle.dump(cheq, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def reloadSim(potential,chkpt_folder, sim_name, time_step):
        with open(f"{chkpt_folder}/chkpts/{sim_name}_{time_step}.pickle", "rb") as handle:
            dmcObj =  pickle.load(handle)
        dmcObj.potential = potential
        return dmcObj