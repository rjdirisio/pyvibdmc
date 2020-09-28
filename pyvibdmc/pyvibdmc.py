"""
This is the main file, which runs the DMC code itself.  To see the basic algorithm in
action, go to self.propagate()
"""
import time

import numpy as np

from .simulation_utilities import *


class DMC_Sim:
    """
    The main object that you will use to run the DMC simulation. You will instantiate an object DMC_SIM(...)
    and then use object.run() to begin the simulation.

    :param sim_name: Simulation name for saving wave functions and checkpointing
    :type sim_name: str
    :param output_folder: The folder where the results will be stored, including wave functions and energies. Abs. or rel. path.
    :type output_folder: str
    :param weighting: Discrete or Continuous weighting DMC.  Continuous means that there are fixed number of walkers
    :type weighting: str
    :param num_walkers: Number of walkers we will start the simulation with
    :type num_walkers: int
    :param num_timesteps: Total time steps we will be propagating the walkers. num_timesteps*delta_t = total time in A.U.
    :type num_timesteps: int
    :param equil_steps: Time steps before we start collecting wave function
    :type equil_steps: int
    :param chkpt_every: How many time steps in between checkpoints (only checkpoints once we reached equil_steps.
    :type chkpt_every: int
    :param desc_wt_time_steps: Number of time steps monitoring the wave function's desc_wt_timeendants.
    :type desc_wt_time_steps: int
    :param atoms: List of atoms for the simulation
    :type atoms: list
    :param delta_t: The length of the time step; how many atomic units of time are you going in one time step.
    :type delta_t: int
    :param potential: Takes in coordinates, gives back energies
    :type potential: function
    :param masses: For feeding in artificial masses in atomic units.  If not, then the atoms param will designate masses
    :type masses: list
    :param start_structures: An initial structure to initialize all your walkers
    :type start_structures: np.ndarray
    :param branch_every: Branch the walkers every x time steps, also known as birth/death
    :type branch_every: int
    :param log_every: Gather simulation information every n time steps. This includes timing of potential call, max/min of weight and energy of ensemble, and number of births/deaths or branching
    :type log_every: int
    :param cur_timestep: The current time step, should be zero unless you are restarting
    :type cur_timestep: int
    :param DEBUG_alpha: The number that will be used instead of 1/(2*delta_t) for alpha.
    :type DEBUG_alpha: float
    :param DEBUG_save_desc_wt_time_tracker: If true, will save the array that keeps track of births/deaths during desc_wt_time weighting. Will save as a binary .npy file (see numpy documentation)
    :type DEBUG_save_desc_wt_time_tracker: bool
    :param DEBUG_save_training_every: If true, will collect coordinates and energies every n time steps
    :type DEBUG_save_training_every: int
    """

    def __init__(self,
                 sim_name="DMC_Sim",
                 output_folder="exSimResults",
                 weighting='discrete',
                 num_walkers=10000,
                 num_timesteps=20000,
                 equil_steps=2000,
                 chkpt_every=1000,
                 wfn_every=1000,
                 desc_wt_time_steps=100,
                 atoms=[],
                 delta_t=5,
                 potential=None,
                 masses=None,
                 start_structures=None,
                 branch_every=1,
                 log_every=100,
                 cur_timestep=0,
                 DEBUG_alpha=None,
                 DEBUG_save_desc_wt_time_tracker=None,
                 DEBUG_save_training_every=None
                 ):
        self.atoms = atoms
        self.sim_name = sim_name
        self.output_folder = output_folder
        self.num_walkers = num_walkers
        self.num_timesteps = num_timesteps
        self.potential = potential.getpot
        self.weighting = weighting.lower()
        self.desc_wt_time_steps = desc_wt_time_steps
        self.branch_every = branch_every
        self.delta_t = delta_t
        self.start_structures = start_structures
        self.masses = masses
        self.equil_steps = equil_steps
        self.chkpt_every = chkpt_every
        self.wfn_every = wfn_every
        self.log_every = log_every
        self.cur_timestep = cur_timestep
        self._deb_training_every = DEBUG_save_training_every
        self._deb_desc_wt_time_tracker = DEBUG_save_desc_wt_time_tracker
        self._deb_alpha = DEBUG_alpha
        self._initialize()

    def _initialize(self):
        """
        Initialization of all the arrays and internal simulation parameters.
        """
        # Initialize the rest of the (private) variables needed for the simulation
        # Arrays used to mark important events throughout the simulation
        self._prop_steps = np.arange(self.cur_timestep, self.num_timesteps)
        self._branch_step = np.arange(self.cur_timestep, self.num_timesteps + self.branch_every, self.branch_every)
        self._chkpt_step = np.arange(self.equil_steps, self.num_timesteps + self.chkpt_every, self.chkpt_every)
        self._wfn_save_step = np.arange(self.equil_steps, self.num_timesteps + self.wfn_every, self.wfn_every)
        self._desc_wt_save_step = self._wfn_save_step + self.desc_wt_time_steps
        if self._deb_training_every is not None:
            self.deb_train_save_step = np.arange(self.cur_timestep, self.num_timesteps + self._deb_training_every,
                                                 self._deb_training_every)
        else:
            self.deb_train_save_step = []
        self._log_steps = np.arange(self.cur_timestep, self.num_timesteps, self.log_every)
        # Arrays that carry data throughout the simulation
        self._who_from = None  # desc_wt_timeendant weighting doesn't happen right away, no need to init
        self._walker_pots = None  # will get returned from potential function
        self._vref_vs_tau = np.zeros(self.num_timesteps)
        self._pop_vs_tau = np.zeros(self.num_timesteps)

        # Set up coordinates array
        if self.start_structures is None:
            raise Exception("Please supply a starting structure for your chemical system.")
        elif len(self.start_structures.shape) != 3:
            raise Exception("Start structure must have format (nxmxd), where n = 1 or num_walkers, m = num atoms, "
                            "d = dims")
        elif self.start_structures.shape[0] == 1:
            self._walker_coords = np.repeat(self.start_structures,
                                            self.num_walkers, axis=0)
        elif self.start_structures.shape[0] == self.num_walkers:
            self._walker_coords = self.start_structures
        else:
            print("WARNING: NUMBER OF STARTING GEOMETRIES DOES NOT EQUAL NUM_WALKERS VARIABLE. MAKE SURE THIS IS"
                  "INTENTIONAL.")
            self._walker_coords = self.start_structures

        # Set up masses and sigmas
        if self.masses is None:
            self.masses = np.array([Constants.mass(a) for a in self.atoms])
        if len(self.masses) != len(self.atoms):
            raise Exception("Your number of atoms list does not match your number of self.masses you provided.")
        self._atm_nums = get_atomic_num(self.atoms)

        # Constants for simulation
        self._sigmas = np.sqrt(self.delta_t / self.masses)
        if self._deb_alpha is None:
            self._alpha = 1.0 / (2.0 * self.delta_t)
        else:
            self._alpha = self._deb_alpha

        # Where to save the data
        FileManager.create_filesystem(self.output_folder)
        self._logger = SimLogger(f"{self.output_folder}/{self.sim_name}.log", overwrite=True)

        # Weighting technique
        if self.weighting == 'continuous':
            self._cont_wts = np.ones(self.num_walkers)
            self._thresh = 1 / self.num_walkers  # continuous weighting threshold
        else:
            self._cont_wts = None

    def birth_or_death(self, desc_wt_time):
        """
        Chooses whether or not the walker made a bad enough random walk to be removed from the simulation.
        For discrete weighting, this leads to removal or duplication of the walkers.  For continuous, this leads
        to an update of the weights and a potential branching of a large weight walker to the smallest one
        :param desc_wt_time: A boolean that checks if desc_wt_timeendent weighting should be occuring right now.
        :type desc_wt_time: bool
        :param log_it: A boolean that will log births and deaths.
        :type log_it: bool
        :return: Updated Continus Weights , the "who from" array for desc_wt_timeendent weighting, walker coords, and pot vals.
         """
        if self.weighting == 'discrete':
            rand_nums = np.random.random(len(self._walker_coords))

            death_mask = np.logical_or((1 - np.exp(-1. * (self._walker_pots - self._vref) * self.delta_t)) < rand_nums,
                                       self._walker_pots < self._vref)

            num_deaths = len(self._walker_coords) - np.sum(death_mask)

            self._walker_coords = self._walker_coords[death_mask]
            self._walker_pots = self._walker_pots[death_mask]
            rand_nums = rand_nums[death_mask]
            if desc_wt_time:
                self._who_from = self._who_from[death_mask]

            exTerm = np.exp(-1. * (self._walker_pots - self._vref) * self.delta_t) - 1
            ct = 1
            num_births = 0
            while np.amax(exTerm) > 0.0:
                if ct != 1:
                    rand_nums = np.random.random(
                        len(self._walker_coords))  # get new random numbers for probability of birth
                birth_mask = np.logical_and(exTerm > rand_nums,
                                            self._walker_pots < self._vref)

                num_births += np.sum(birth_mask)

                self._walker_coords = np.concatenate((self._walker_coords,
                                                      self._walker_coords[birth_mask]))
                self._walker_pots = np.concatenate((self._walker_pots,
                                                    self._walker_pots[birth_mask]))
                if desc_wt_time:
                    self._who_from = np.concatenate((self._who_from,
                                                     self._who_from[birth_mask]))
                exTerm = np.exp(-1. * (self._walker_pots - self._vref) * self.delta_t) - (1 + ct)
                ct += 1
                return num_births, num_deaths
        else:
            self._cont_wts = self._cont_wts * np.exp(-1.0 * (self._walker_pots - self._vref) * self.delta_t)
            kill_mark = np.where(self._cont_wts < self._thresh)[0]
            num_branched = len(kill_mark)
            for walker in kill_mark:
                maxWalker = np.argmax(self._cont_wts)
                self._walker_coords[walker] = np.copy(self._walker_coords[maxWalker])
                self._walker_pots[walker] = np.copy(self._walker_pots[maxWalker])
                if desc_wt_time:
                    self._who_from[walker] = self._who_from[maxWalker]
                self._cont_wts[maxWalker] /= 2.0
                self._cont_wts[walker] = np.copy(self._cont_wts[maxWalker])
                return num_branched

    def moveRandomly(self):
        """
        The random displacement of each of the coordinates of each of the walkers, done in a vectorized fashion. Displaces self._walker_coords
        """
        disps = np.random.normal(0.0,
                                 self._sigmas,
                                 size=np.shape(self._walker_coords.transpose(0, 2, 1))).transpose(0, 2, 1)
        self._walker_coords += disps

    def calc_vref(self):  # Use potential of all walkers to calculate self._vref
        """
        Use the energy of all walkers to calculate self._vref with a correction for the fluctuation in the population or weight. Updates Vref
        """
        if self.weighting == 'discrete':
            Vbar = np.average(self._walker_pots)
            correction = (len(self._walker_pots) - self.num_walkers) / self.num_walkers
        else:
            Vbar = np.average(self._walker_pots, weights=self._cont_wts)
            correction = (np.sum(self._cont_wts - np.ones(self.num_walkers))) / self.num_walkers
        self._vref = Vbar - (self._alpha * correction)

    def calc_desc_wt_time_weights(self, desc_wts):
        """
        At the end of desc_wt_timeendent weighting, count up which walkers came from other walkers (desc_wt_timeendants)

        :param desc_wts: The array in which the desc_wt_timeendent weights will be stored
        """
        if self.weighting == 'discrete':
            unique, counts = np.unique(self._who_from, return_counts=True)
            desc_wts[unique] = counts
        else:
            for q in range(len(self._cont_wts)):
                desc_wts[q] = np.sum(self._cont_wts[self._who_from == q])
        return desc_wts

    def update_sim_arrays(self, prop_step):
        """
        :param prop_step:  the time step that we are currently on
        """
        self._vref_vs_tau[prop_step] = self._vref
        if self.weighting == 'discrete':
            self._pop_vs_tau[prop_step] = len(self._walker_coords)
        else:
            self._pop_vs_tau[prop_step] = np.sum(self._cont_wts)

    def propagate(self):
        """
             The main DMC loop.
             1. Move Randomly
             2. Calculate the Potential Energy
             3. Birth/Death
             4. Update Vref
             Additionally, checks when the wavefunction has hit a point where it should save / do desc_wt_timeendent weighting.
         """
        desc_wt = False
        for prop_step in self._prop_steps:
            self.cur_timestep = prop_step

            # Check if prop_step is at a special point in simulation
            if prop_step in self._log_steps:
                self._logger.write_ts(prop_step)

            if prop_step in self.deb_train_save_step and prop_step != 0:
                print(prop_step, self.deb_train_save_step)
                print(self._walker_coords.shape, self._walker_pots)
                SimArchivist.save_h5(fname=f"{self.output_folder}/{self.sim_name}_training_{prop_step}ts.hdf5",
                                     keyz=['coords', 'pots'], valz=[self._walker_coords, self._walker_pots])

            # If we are at a checkpoint
            if prop_step in self._chkpt_step:
                self._logger.write_chkpt(prop_step)
                self._logger = None
                SimArchivist.chkpt(self, prop_step)
                self._logger = SimLogger(f"{self.output_folder}/{self.sim_name}.log")

            # If we are at a spot to begin desc_wt_timeendant weighting
            if prop_step in self._wfn_save_step:
                self._logger.write_wfn_save(prop_step)
                desc_wts = np.zeros(len(self._walker_coords))
                parent = np.copy(self._walker_coords)
                self._who_from = np.arange(len(self._walker_coords))
                desc_wt = True

            # If desc_wt_time weighting is over, save the wfn and weights
            if prop_step in self._desc_wt_save_step:
                self._logger.write_desc_wt(prop_step)
                desc_wt = False
                desc_wts = self.calc_desc_wt_time_weights(desc_wts)
                vref_cm = Constants.convert(self._vref_vs_tau, 'wavenumbers', to_AU=False)
                SimArchivist.save_h5(
                    fname=f"{self.output_folder}/wfns/{self.sim_name}_wfn_{prop_step - self.desc_wt_time_steps}ts.hdf5",
                    keyz=['coords', 'desc_wt_time_weights', 'desc_wt_time_time', 'atomic_nums', 'vref_vs_tau'],
                    valz=[parent, desc_wts, self.desc_wt_time_steps, self._atm_nums, vref_cm])
                if self._deb_desc_wt_time_tracker:
                    np.save(
                        f"{self.output_folder}/wfns/{self.sim_name}_desc_wt_time_tracker_{prop_step - self.desc_wt_time_steps}ts",
                        self._who_from)

            # 1. Move Randomly
            self.moveRandomly()

            # 2. Calculate the Potential Energy
            if prop_step in self._log_steps:
                self._walker_pots, pot_time = self.potential(self._walker_coords, timeit=True)
                self._logger.write_pot_time(prop_step, pot_time)
            else:
                self._walker_pots = self.potential(self._walker_coords)

            # First time step exception, calc vref early
            if prop_step == self._prop_steps[0]:
                self.calc_vref()

            # 3. Birth/Death
            if prop_step in self._branch_step:
                ensemble_changes = self.birth_or_death(desc_wt)
                if prop_step in self._log_steps:
                    self._logger.write_branching(prop_step, self.weighting, ensemble_changes)
            else:
                if self.weighting == 'continuous':  # update weights but no branching
                    self._cont_wts *= np.exp(-1.0 * (self._walker_pots - self._vref) * self.delta_t)

            # 4. Update Vref.
            self.calc_vref()
            self.update_sim_arrays(prop_step)

    def run(self):
        """This function calls propagate and saves simulation results"""
        self.propagate()
        _vrefCM = Constants.convert(self._vref_vs_tau, "wavenumbers", to_AU=False)
        print('Approximate ZPE', np.average(_vrefCM[len(_vrefCM) // 4:]))
        ts = np.arange(len(_vrefCM))
        SimArchivist.save_h5(fname=f"{self.output_folder}/{self.sim_name}_sim_info.hdf5",
                             keyz=['vref_vs_tau', 'pop_vs_tau', 'atomic_nums'],
                             valz=[np.column_stack((ts, _vrefCM)), np.column_stack((ts, self._pop_vs_tau)),
                                   self._atm_nums])

    def __deepcopy__(self, memodict={}):
        """
        Helper internal copying system for checkpointing.
        Here to ensure that the potential is not pickled.
        """
        cls = self.__class__
        res = cls.__new__(cls)
        memodict[id(self)] = res
        for k, v in self.__dict__.items():
            if k != 'potential':
                setattr(res, k, copy.deepcopy(v, memodict))
        return res


def DMC_Restart(potential,
                time_step,
                chkpt_folder="exSimulation_results/",
                sim_name='DMC_Sim'):
    dmc_sim = SimArchivist.reload_sim(chkpt_folder, sim_name, time_step)
    dmc_sim.cur_timestep = time_step
    dmc_sim.potential = potential
    dmc_sim._initialize()
    FileManager.delete_future_checkpoints(chkpt_folder, sim_name, time_step)
    return dmc_sim


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print('hi')
