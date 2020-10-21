import os
import glob as glob


class FileManager:
    """Helping with creating or deleting files as necessary throughout the simulation"""

    @staticmethod
    def delete_future_checkpoints(chkpt_folder, sim_name, time_step):
        """
        When restarting and testing, this helper classs takes deletes specified simulation checkpoints and wavefunctions
        """
        pickles = glob.glob(f'{chkpt_folder}/chkpts/{sim_name}*.pickle')
        wfns = glob.glob(f'{chkpt_folder}/wfns/{sim_name}*.hdf5')
        pickles.sort()
        wfns.sort()
        ts = [int(p.split('_')[-1].split('.')[0]) for p in pickles]
        ts_wfn = [int(w.split('_')[-1].split('ts')[0]) for w in wfns]
        for pickN, pickTime in enumerate(ts):
            if pickTime > time_step:  # if the pickle file is older than the time step you are restarting from, delete it.
                os.remove(pickles[pickN])
        for wfnN, wfnTime in enumerate(ts_wfn):
            if wfnTime > time_step:
                os.remove(wfns[wfnN])

    @staticmethod
    def create_filesystem(output_folder):
        """
        Creates folders and subfolders to house the DMC simulation results and checkpoints.
        """
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            os.makedirs(output_folder + '/chkpts')
            os.makedirs(output_folder + '/wfns')
