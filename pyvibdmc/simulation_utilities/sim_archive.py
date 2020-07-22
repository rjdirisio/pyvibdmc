import h5py
import numpy as np
import pickle

class SimArchivist:
    """A utility class for saving wave functions, checkpoint files, and reloading DMC sims"""
    @classmethod
    def saveH5(cls,fname, keyz, valz):
        with h5py.File(fname, 'w') as hf:
            for key, val in zip(keyz, valz):
                dset = hf.create_dataset(key,
                                     data=val)

    @classmethod
    def chkpt(cls,dmcObj,propStep):
        with open(f'{dmcObj.outputFolder}/chkpt_{dmcObj.simName}_{str(propStep)}.pickle', 'wb') as handle:
            pickle.dump(dmcObj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def reloadSim(cls,ckptFolder,simName,timeStep):
        with open(f'{ckptFolder}/chkpt_{simName}_{str(timeStep)}.pickle', 'rb') as handle:
            return  pickle.load(handle)
