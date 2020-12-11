import os
import glob as glob


class FileManager:
    """Helping with creating or deleting files as necessary throughout the simulation"""

    @staticmethod
    def delete_future_checkpoints(sim_folder, sim_name, time_step):
        """
        When restarting and testing, this helper classs takes deletes specified simulation checkpoints and wavefunctions
        """
        pickles = glob.glob(f'{sim_folder}/chkpts/{sim_name}_*.pickle')
        wfns = glob.glob(f'{sim_folder}/wfns/{sim_name}_*.hdf5')
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
    def delete_older_checkpoints(sim_folder, sim_name, time_step):
        """
        When checkpointing, this helper takes deletes checkpoints along the way
        """
        pickles = glob.glob(f'{sim_folder}/chkpts/{sim_name}*.pickle')
        pickles.sort()
        ts = [int(p.split('_')[-1].split('.')[0]) for p in pickles]
        for pickN, pickTime in enumerate(ts):
            if pickTime < time_step:  # if the pickle file is older than the time step you are restarting from, delete it.
                os.remove(pickles[pickN])


    @staticmethod
    def create_filesystem(output_folder):
        """
        Creates folders and subfolders to house the DMC simulation results and checkpoints.
        """
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            os.makedirs(output_folder + '/chkpts')
            os.makedirs(output_folder + '/wfns')
        else:
            # Output folder already exists. Make sure subfolders exist.
            if not os.path.isdir(output_folder + '/chkpts'):
                os.makedirs(output_folder + '/chkpts')
            if not os.path.isdir(output_folder + '/wfns'):
                os.makedirs(output_folder + '/wfns')

