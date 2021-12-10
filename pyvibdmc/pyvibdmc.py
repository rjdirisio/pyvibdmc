"""
This is the main file, which runs the DMC code itself.  To see the basic algorithm in
action, go to self.propagate()
"""
import numpy as np
import time, copy

from .simulation_utilities.file_manager import *
from .simulation_utilities.sim_archive import *
from .simulation_utilities.Constants import *
from .simulation_utilities.sim_logger import *
from .simulation_utilities.imp_samp_manager import *
from .simulation_utilities.imp_samp import *

__all__ = ['DMC_Sim', 'dmc_restart']


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
    :param desc_wt_steps: Number of time steps monitoring the wave function's desc_wt_timeendants.
    :type desc_wt_steps: int
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
    :param cont_wt_thresh: If a weight goes past this bound, branch it.  If only supplied one number, it will use that for the lower bound.
    :type cont_wt_thresh: list of floats
    :param DEBUG_alpha: The number that will be used instead of 1/(2*delta_t) for alpha.
    :type DEBUG_alpha: float
    :param DEBUG_save_desc_wt_tracker: If true, will save the array that keeps track of births/deaths during desc_wt_time weighting. Will save as a binary .npy file (see numpy documentation)
    :type DEBUG_save_desc_wt_tracker: bool
    :param DEBUG_save_training_every: If true, will collect coordinates and energies every n time steps
    :type DEBUG_save_training_every: int
    :param DEBUG_mass_change: Dictionary that will scale the mass every change_every steps. The scaling factor_per_change can be an array or an int/float
    "type DEBUG_mass_change: dict
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
                 desc_wt_steps=100,
                 atoms=[],
                 delta_t=1,
                 potential=None,
                 masses=None,
                 start_structures=None,
                 start_cont_wts=None,
                 branch_every=1,
                 log_every=1,
                 cur_timestep=0,
                 cont_wt_thresh=None,
                 imp_samp=None,
                 imp_samp_oned=False,
                 adiabatic_dmc=None,
                 DEBUG_alpha=None,
                 DEBUG_save_desc_wt_tracker=None,
                 DEBUG_save_training_every=None,
                 DEBUG_save_before_bod=False,
                 DEBUG_mass_change=None
                 ):
        self.atoms = atoms
        self.sim_name = sim_name
        self.output_folder = output_folder
        self.num_walkers = num_walkers
        self.num_timesteps = num_timesteps
        self.potential_info = vars(potential)
        self.potential = potential.getpot
        self.weighting = weighting.lower()
        self.desc_wt_time_steps = desc_wt_steps
        self.branch_every = branch_every
        self.delta_t = delta_t
        self.start_structures = start_structures
        self.start_cont_wts = start_cont_wts
        self.masses = masses
        self.equil_steps = equil_steps
        self.chkpt_every = chkpt_every
        self.wfn_every = wfn_every
        self.log_every = log_every
        self.cur_timestep = cur_timestep
        self.cont_wt_thresh = cont_wt_thresh
        self.impsamp_manager = imp_samp
        self.imp1d = imp_samp_oned
        self.adiabatic_dmc = adiabatic_dmc
        self._deb_training_every = DEBUG_save_training_every
        self._deb_save_before_bod = DEBUG_save_before_bod
        self._deb_desc_wt_tracker = DEBUG_save_desc_wt_tracker
        self._deb_alpha = DEBUG_alpha
        self._deb_mass_change = DEBUG_mass_change
        self._initialize()

    def _initialize(self):
        """
        Initialization of all the arrays and internal simulation parameters.
        """
        # Initialize the rest of the (private) variables needed for the simulation
        # Arrays used to mark important events throughout the simulation
        self._prop_steps = np.arange(0, self.num_timesteps)
        self._branch_step = np.arange(0, self.num_timesteps + self.branch_every, self.branch_every)
        self._chkpt_step = np.arange(self.chkpt_every, self.num_timesteps + self.chkpt_every, self.chkpt_every)
        self._wfn_save_step = np.arange(self.equil_steps, self.num_timesteps + self.wfn_every, self.wfn_every)
        self._desc_wt_save_step = self._wfn_save_step + self.desc_wt_time_steps
        if self._deb_training_every is not None:
            self.deb_train_save_step = np.arange(0, self.num_timesteps + self._deb_training_every,
                                                 self._deb_training_every)
        else:
            self.deb_train_save_step = []
        self._log_steps = np.arange(0, self.num_timesteps, self.log_every)
        # Arrays that carry data throughout the simulation
        self._who_from = None  # weighting doesn't happen right away, no need to init
        self._walker_pots = None  # will get returned from potential function
        self._vref_vs_tau = np.zeros(self.num_timesteps)
        self._pop_vs_tau = np.zeros(self.num_timesteps)

        # Set up coordinates array
        if self.start_structures is None:
            raise Exception("Please supply a starting structure for your chemical system.")
        elif len(self.start_structures.shape) != 3:
            raise Exception("Start structure must have format (n,m,d), where n = 1 or num_walkers, m = num atoms, "
                            "d = dimensions (usually 3)")
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
        if type(self.atoms) is not list:
            self.atoms = [self.atoms]

        # pull masses from atoms
        if self.masses is None:
            # reduced mass - 1D problem
            if '-' in self.atoms[0]:
                self.masses = np.array([Constants.reduced_mass(self.atoms[0])])
                self._atm_nums = get_atomic_num(self.atoms[0].split('-'))
            # regular molecules
            else:
                self.masses = np.array([Constants.mass(a) for a in self.atoms])
                self._atm_nums = get_atomic_num(self.atoms)
        # custom masses
        elif isinstance(self.masses, (int, float)):
            self.masses = np.array([self.masses])
            self._atm_nums = [-1]
        elif isinstance(self.masses, (list, np.ndarray)):
            self.masses = np.array(self.masses)
            self._atm_nums = get_atomic_num(self.atoms)

        if len(self.masses) != len(self.atoms):
            raise Exception("Your number of atoms list does not match your number of masses you provided.")
        if self._walker_coords.shape[1] != len(self.atoms):
            raise Exception("Your number of atoms list does not match the shape of your walkers.")

        # Constants for simulation
        self._sigmas = np.sqrt(self.delta_t / self.masses)
        if self._deb_alpha is None:
            self._alpha = 1.0 / (2.0 * self.delta_t)
        else:
            self._alpha = self._deb_alpha

        # Where to save the data
        FileManager.create_filesystem(self.output_folder)
        if self.cur_timestep == 0:
            rest = True
        else:
            rest = False
        self._logger = SimLogger(f"{self.output_folder}/{self.sim_name}_log.txt", overwrite=rest)

        # Weighting technique
        if self.weighting == 'continuous':
            self._thresh_upper = None
            if self.start_cont_wts is not None:
                self._cont_wts = self.start_cont_wts
            else:
                self._cont_wts = np.ones(self.num_walkers)
            if self.cont_wt_thresh is None:
                self._thresh_lower = 1 / self.num_walkers  # default continuous weighting threshold
            elif isinstance(self.cont_wt_thresh, int) or isinstance(self.cont_wt_thresh, float):
                self._thresh_lower = self.cont_wt_thresh
            elif isinstance(self.cont_wt_thresh, list):
                if len(self.cont_wt_thresh) == 2:
                    self._thresh_lower = self.cont_wt_thresh[0]
                    self._thresh_upper = self.cont_wt_thresh[1]
                elif len(self.cont_wt_thresh) == 1:
                    self._thresh_lower = self.cont_wt_thresh[0]
            else:
                raise ValueError("Invalid input for continuous weight threshold")
        else:
            self._cont_wts = None
        self._desc_wt = False

        # Mass change throughout simulation
        if self._deb_mass_change is not None:
            change_every = self._deb_mass_change['change_every']
            self._mass_change_steps = np.arange(0, self.num_timesteps, change_every)[1:]
            self._factor_per_change = self._deb_mass_change['factor_per_change']  # number or numpy array
            if isinstance(self._factor_per_change, int) or isinstance(self._factor_per_change, float):
                self._factor_per_change = np.repeat(self._factor_per_change, len(self._mass_change_steps))
            if len(self._factor_per_change) != len(self._mass_change_steps):
                raise ValueError("Number of mass change steps must be equal to mass changes you provide.")
            self._mass_counter = 0
        else:
            self._mass_change_steps = []

        # Check for if population fluctuates too much
        self._pop_thresh = [self.num_walkers - self.num_walkers * 0.5,
                            self.num_walkers + self.num_walkers * 0.5]

        if 'num_mpi' in self.potential_info.keys():
            """This is an MPI potential. Run it through once to initialize it"""
            self.potential(self._walker_coords)

        if self.impsamp_manager is not None:
            self.imp_info = vars(self.impsamp_manager)
            if self.delta_t != 1:
                print("WARNING! Using DT>1 for Imp Samp. Make sure this is not a mistake!!!!")
                # raise ValueError("Delta tau cannot be anything but 1 for importance sampling DMC!!!!")
            self.f_x = None
            self.psi_1 = None
            self.psi_sec_der = None
            # Useful variables to have for importance sampling
            self.inv_masses_trip = (1 / np.repeat(self.masses, 3)).reshape(len(self.masses), 3)[np.newaxis, ...]
            self.sigma_trip = np.repeat(self._sigmas, 3).reshape(len(self.masses), 3)[np.newaxis, ...]
            self.impsamp = ImpSamp(self.impsamp_manager)
            self.eff_ts = np.zeros(len(self._prop_steps))

            if self.imp1d:
                # No xyz just one mass and one sigma
                self.sigma_trip = self._sigmas
                self.inv_masses_trip = (1 / self.masses)[np.newaxis]

        if self.adiabatic_dmc is not None:
            # adia_dict = {'initial_lambda': -200,
            #              'lambda_change': 0.05,
            #              'equil_time': 1000,
            #              'observable_func': func,
            #              }
            ad_lam = self.adiabatic_dmc['initial_lambda']
            ad_lam_dx = self.adiabatic_dmc['lambda_change']
            ad_eq_time = self.adiabatic_dmc['equil_time']
            self.ad_obs_func = self.adiabatic_dmc['observable_func']  # user defined
            a = np.zeros(ad_eq_time)
            b = np.arange(ad_lam,
                          ad_lam + (ad_lam_dx * (self.num_timesteps - ad_eq_time)),
                          ad_lam_dx)
            self.ad_lam_array = np.concatenate((a,b))

    def _init_restart(self, add_ts, impsamp):
        """ Reset internal DMC parameters based on additional time steps one wants to run for"""
        self.num_timesteps = self.num_timesteps + add_ts
        self._branch_step = np.arange(0, self.num_timesteps + self.branch_every, self.branch_every)
        self._chkpt_step = np.arange(self.chkpt_every, self.num_timesteps + self.chkpt_every, self.chkpt_every)
        self._wfn_save_step = np.arange(self.equil_steps, self.num_timesteps + self.wfn_every, self.wfn_every)
        self._desc_wt_save_step = self._wfn_save_step + self.desc_wt_time_steps
        if self._deb_training_every is not None:
            self.deb_train_save_step = np.arange(0, self.num_timesteps + self._deb_training_every,
                                                 self._deb_training_every)
        else:
            self.deb_train_save_step = []
        self._log_steps = np.arange(0, self.num_timesteps, self.log_every)
        self._vref_vs_tau = np.concatenate((self._vref_vs_tau, np.zeros(add_ts)))
        self._pop_vs_tau = np.concatenate((self._pop_vs_tau, np.zeros(add_ts)))

        self._prop_steps = np.arange(self.cur_timestep, self.num_timesteps)

        if impsamp is not None:
            self.impsamp_manager = impsamp
            self.imp1d = False
            self.imp_info = vars(self.impsamp_manager)
            if self.delta_t != 1:
                raise ValueError("Delta tau cannot be anything but 1 for importance sampling DMC!!!!")
            self.f_x = None
            self.psi_1 = None
            self.psi_sec_der = None
            # Useful variables to have for importance sampling
            self.inv_masses_trip = (1 / np.repeat(self.masses, 3)).reshape(len(self.masses), 3)[np.newaxis, ...]
            self.sigma_trip = np.repeat(self._sigmas, 3).reshape(len(self.masses), 3)[np.newaxis, ...]
            self.impsamp = ImpSamp(self.impsamp_manager)
        else:
            self.impsamp_manager = None
        self.adiabatic_dmc = None

    def _branch(self, walkers_below):
        """
        Helper class that actually does the branching
        """
        for walker in walkers_below:
            max_walker = np.argmax(self._cont_wts)
            self._walker_coords[walker] = np.copy(self._walker_coords[max_walker])
            self._walker_pots[walker] = np.copy(self._walker_pots[max_walker])

            if self._desc_wt:
                self._who_from[walker] = self._who_from[max_walker]
            self._cont_wts[max_walker] /= 2.0
            self._cont_wts[walker] = np.copy(self._cont_wts[max_walker])
            if self.impsamp_manager is not None:
                self.f_x[walker] = np.copy(self.f_x[max_walker])
                self.psi_1[walker] = np.copy(self.psi_1[max_walker])
                self.psi_sec_der[walker] = np.copy(self.psi_sec_der[max_walker])

    @property
    def vref_vs_tau(self):
        """Returns the vref array, including zeros from initialization"""
        # vref_wvn = Constants.convert(self._vref_vs_tau[:self.cur_timestep], "wavenumbers", to_AU=False)
        vref_wvn = self._vref_vs_tau[:self.cur_timestep]
        return np.column_stack((np.arange(len(vref_wvn)), vref_wvn))

    @property
    def walkers(self):
        if self.weighting == 'continuous':
            return self._walker_coords, self._cont_wts
        else:
            return self._walker_coords

    def update_effetive_timestep(self):
        dt = self.delta_t * self.dt_factor
        if self.cur_timestep == 0:
            self.eff_ts[self.cur_timestep] = dt
        else:
            self.eff_ts[self.cur_timestep] = self.eff_ts[self.cur_timestep - 1] + dt
        return dt

    def birth_or_death(self):
        """
        Chooses whether or not the walker made a bad enough random walk to be removed from the simulation.
        For discrete weighting, this leads to removal or duplication of the walkers.  For continuous, this leads
        to an update of the weights and a potential branching of a large weight walker to the smallest one
        :return: Updated Continus Weights , the "who from" array for desc_wt_timeendent weighting, walker coords, and pot vals.
         """
        dt = self.delta_t
        if self.impsamp_manager is not None:
            dt = self.update_effetive_timestep()

        if self.weighting == 'discrete':
            rand_nums = np.random.random(len(self._walker_coords))

            death_mask = np.logical_or((1 - np.exp(-1. * (self._walker_pots - self._vref) * dt)) < rand_nums,
                                       self._walker_pots < self._vref)

            num_deaths = len(self._walker_coords) - np.sum(death_mask)

            self._walker_coords = self._walker_coords[death_mask]
            self._walker_pots = self._walker_pots[death_mask]
            rand_nums = rand_nums[death_mask]
            if self.impsamp_manager is not None:
                self.f_x = self.f_x[death_mask]
                self.psi_1 = self.psi_1[death_mask]
                self.psi_sec_der = self.psi_sec_der[death_mask]
            if self._desc_wt:
                self._who_from = self._who_from[death_mask]

            exp_term = np.exp(-1. * (self._walker_pots - self._vref) * dt) - 1
            ct = 1
            num_births = 0
            while np.amax(exp_term) > 0.0:
                if ct != 1:
                    rand_nums = np.random.random(
                        len(self._walker_coords))  # get new random numbers for probability of birth
                birth_mask = np.logical_and(exp_term > rand_nums,
                                            self._walker_pots < self._vref)

                num_births += np.sum(birth_mask)

                self._walker_coords = np.concatenate((self._walker_coords,
                                                      self._walker_coords[birth_mask]))
                self._walker_pots = np.concatenate((self._walker_pots,
                                                    self._walker_pots[birth_mask]))
                if self.impsamp_manager is not None:
                    self.f_x = np.concatenate((self.f_x, self.f_x[birth_mask]))
                    self.psi_1 = np.concatenate((self.psi_1, self.psi_1[birth_mask]))
                    self.psi_sec_der = np.concatenate((self.psi_sec_der, self.psi_sec_der[birth_mask]))

                if self._desc_wt:
                    self._who_from = np.concatenate((self._who_from,
                                                     self._who_from[birth_mask]))
                exp_term = np.exp(-1. * (self._walker_pots - self._vref) * dt) - (1 + ct)
                ct += 1
            return num_births, num_deaths, len(self._walker_pots)
        else:
            self._cont_wts *= np.exp(-1.0 * (self._walker_pots - self._vref) * dt)

            # branch if weights are too low
            kill_mark = np.where(self._cont_wts < self._thresh_lower)[0]
            self._branch(kill_mark)

            # branch more if there are weights that are too high
            if self._thresh_upper is not None:
                num_above_thresh = np.sum(
                    self._cont_wts > self._thresh_upper)  # now, see if any weights are still too big
                kill_mark_upper = np.argpartition(self._cont_wts, num_above_thresh)[
                                  :num_above_thresh]  # get the num_above_thresh smallest wts
                self._branch(kill_mark_upper)
            else:
                kill_mark_upper = []

            # logging info
            num_branched = len(kill_mark) + len(kill_mark_upper)
            max_weght = np.amax(self._cont_wts)
            min_weght = np.amin(self._cont_wts)

            return num_branched, max_weght, min_weght

    def move_randomly(self):
        """
        The random displacement of each of the coordinates of each of the walkers, done in a vectorized fashion. Displaces self._walker_coords
        """
        disps = np.random.normal(0.0,
                                 self._sigmas,
                                 size=np.shape(self._walker_coords.transpose(0, 2, 1))).transpose(0, 2, 1)
        self._walker_coords += disps

    def imp_move_randomly(self):
        """
        The random displacement of each of the coordinates of each of the walkers, done in a vectorized fashion. Displaces self._walker_coords
        """
        if (self.f_x is None or self.psi_1 is None):
            self.f_x, self.psi_1, self.psi_sec_der = self.impsamp.drift(self._walker_coords)

        disps = np.random.normal(0.0,
                                 self._sigmas,
                                 size=np.shape(self._walker_coords.transpose(0, 2, 1))).transpose(0, 2, 1)
        # The actual term added to cartesian coords
        d_x = self.inv_masses_trip * self.f_x  # The actual term added to cartesian coords
        displaced_cds = self._walker_coords + disps + d_x * self.delta_t

        f_y, psi_2, psi_sec_der_disp = self.impsamp.drift(displaced_cds)
        d_y = self.inv_masses_trip * f_y  # The actual term added to cartesian coords

        met_nums = self.impsamp.metropolis(sigma_trip=self.sigma_trip,
                                           trial_x=self.psi_1,
                                           trial_y=psi_2,
                                           disp_x=self._walker_coords,
                                           disp_y=displaced_cds,
                                           D_x=d_x,
                                           D_y=d_y,
                                           dt=self.delta_t)
        randos = np.random.random(size=len(self._walker_coords))
        accept = np.argwhere(met_nums > randos)
        self.dt_factor = len(accept) / len(self._walker_coords)
        self._walker_coords[accept] = displaced_cds[accept]
        self.f_x[accept] = f_y[accept]
        self.psi_1[accept] = psi_2[accept]
        self.psi_sec_der[accept] = psi_sec_der_disp[accept]

        num_rejctions = len(self._walker_coords) - len(accept)
        return num_rejctions

    def calc_vref(self):  # Use potential of all walkers to calculate self._vref
        """
        Use the energy of all walkers to calculate self._vref with a correction for the fluctuation in the population or weight. Updates Vref
        """
        if self.weighting == 'discrete':
            v_bar = np.average(self._walker_pots)
            correction = (len(self._walker_pots) - self.num_walkers) / self.num_walkers
        else:
            v_bar = np.average(self._walker_pots, weights=self._cont_wts)
            correction = (np.sum(self._cont_wts - np.ones(self.num_walkers))) / self.num_walkers
        self._vref = v_bar - (self._alpha * correction)

    def calc_desc_wts(self):
        """
        At the end of desc_wt_timeendent weighting, count up which walkers came from other walkers (desc_wt_timeendants)
        """
        if self.weighting == 'discrete':
            unique, counts = np.unique(self._who_from, return_counts=True)
            self._desc_wts[unique] = counts
        else:
            for q in range(len(self._cont_wts)):
                self._desc_wts[q] = np.sum(self._cont_wts[self._who_from == q])

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
        for prop_step in self._prop_steps:
            self.cur_timestep = prop_step

            # Check for massive birth/death event
            if self.weighting == 'discrete':
                cur_pop = len(self._walker_coords)
                if cur_pop < self._pop_thresh[0] or cur_pop > self._pop_thresh[1]:
                    raise ValueError("Massive walker birth or death event!!!!!!! Dying...")

            # Write simulation attributes whether starting or restarting
            if self.cur_timestep == self._prop_steps[0]:
                self._logger.write_beginning(self.__dict__)

            if prop_step in self._log_steps:
                self._logger.write_ts(prop_step)

            # Check if prop_step is at a special point in simulation

            # If we are at a checkpoint
            if prop_step in self._chkpt_step:
                self._logger.write_chkpt(prop_step)
                self._logger = None
                FileManager.delete_older_checkpoints(self.output_folder,
                                                     self.sim_name,
                                                     prop_step)
                SimArchivist.chkpt(self, prop_step)
                self._logger = SimLogger(f"{self.output_folder}/{self.sim_name}_log.txt")

            # If we are at a spot to begin desc_wt_time
            if prop_step in self._wfn_save_step:
                self._logger.write_wfn_save(prop_step)
                self._desc_wts = np.zeros(len(self._walker_coords))
                self._parent = np.copy(self._walker_coords)
                self._parent_wts = np.copy(self._cont_wts)
                self._who_from = np.arange(len(self._walker_coords))
                self._desc_wt = True
                if self._deb_desc_wt_tracker:
                    desc_wt_history = []

            # If we are at a point to change mass (debug option, scale mass)
            if prop_step in self._mass_change_steps:
                self.masses = self.masses * self._factor_per_change[self._mass_counter]
                self._sigmas = np.sqrt(self.delta_t / self.masses)
                self._mass_counter += 1

            # 1. Move Randomly
            if self.impsamp_manager is None:
                self.move_randomly()
            else:
                start = time.time()
                rejected = self.imp_move_randomly()
                if prop_step in self._log_steps:
                    self._logger.write_rejections(rejected, len(self._walker_coords))
                    self._logger.write_imp_disp_time(time.time() - start)

            # 2. Calculate the Potential Energy
            if prop_step in self._log_steps:
                self._walker_pots, pot_time = self.potential(self._walker_coords, timeit=True)
                maxpot = np.amax(self._walker_pots)
                minpot = np.amin(self._walker_pots)
                avgpot = np.average(self._walker_pots)
                self._logger.write_pot_time(prop_step, pot_time, maxpot, minpot, avgpot)
            else:
                self._walker_pots = self.potential(self._walker_coords)

            # Save training data if it's being collected & collect before bod
            if prop_step in self.deb_train_save_step:
                if self._deb_save_before_bod:
                    print(f'{self._walker_coords.shape} walkers collected')
                    SimArchivist.save_h5(fname=f"{self.output_folder}/{self.sim_name}_training_{prop_step}ts.hdf5",
                                         keyz=['coords', 'pots'], valz=[self._walker_coords, self._walker_pots])

            # If importance sampling, calculate local energy,  which is just adding on local KE
            if self.impsamp_manager is not None:
                local_ke = self.impsamp.local_kin(self.inv_masses_trip, self.psi_sec_der)
                self._walker_pots = self._walker_pots + local_ke
                self._logger.write_local(np.average(self._walker_pots))

            # Uncomment if you want to collect the local energy as well as the potential energy for training data
            # if prop_step in self.deb_train_save_step:
            #     print(f'{self._walker_coords.shape} walkers collected')
            #     SimArchivist.save_h5(fname=f"{self.output_folder}/{self.sim_name}_local_training_{prop_step}ts.hdf5",
            #                          keyz=['coords', 'local'], valz=[self._walker_coords, self._walker_pots])

            # If adiabatic DMC, add on perturbation to the energy for this time step
            if self.adiabatic_dmc is not None:
                this_lam = self.ad_lam_array[prop_step]
                this_w = self.ad_obs_func(self._walker_coords)
                # Average? Or just each walker?
                self._walker_pots = self._walker_pots + this_lam * this_w

            # First time step exception, calc vref early
            if prop_step == self._prop_steps[0]:
                self.calc_vref()

            # 3. Birth/Death, or branching
            if prop_step in self._branch_step:
                ensemble_changes = self.birth_or_death()
                if prop_step in self._log_steps:
                    self._logger.write_branching(prop_step, self.weighting, ensemble_changes)
            else:
                if self.weighting == 'continuous':  # update weights but no branching
                    dt = self.delta_t
                    if self.impsamp_manager is not None:
                        dt = self.update_effetive_timestep()
                    self._cont_wts *= np.exp(-1.0 * (self._walker_pots - self._vref) * dt)

            # Save training data if it's being collected & collect after bod 
            if prop_step in self.deb_train_save_step:
                if not self._deb_save_before_bod:
                    print(f'{self._walker_coords.shape} walkers collected')
                    SimArchivist.save_h5(fname=f"{self.output_folder}/{self.sim_name}_training_{prop_step}ts.hdf5",
                                         keyz=['coords', 'pots'], valz=[self._walker_coords, self._walker_pots])

            # 4. Update Vref.
            self.calc_vref()
            self.update_sim_arrays(prop_step)

            if self._desc_wt and self._deb_desc_wt_tracker:
                self.calc_desc_wts()
                desc_wt_history.append(self._desc_wts)
                self._desc_wts = np.zeros(len(self._desc_wts))

            # If desc_wt_time weighting is over, save the wfn and weights
            if prop_step + 1 in self._desc_wt_save_step:
                self._logger.write_desc_wt(prop_step)
                self._desc_wt = False
                self.calc_desc_wts()
                if self.weighting == 'continuous':
                    SimArchivist.save_h5(
                        fname=f"{self.output_folder}/wfns/{self.sim_name}_wfn_{prop_step + 1 - self.desc_wt_time_steps}ts.hdf5",
                        keyz=['coords', 'desc_wts', 'parent_wts'],
                        valz=[self._parent, self._desc_wts, self._parent_wts])
                else:
                    SimArchivist.save_h5(
                        fname=f"{self.output_folder}/wfns/{self.sim_name}_wfn_{prop_step + 1 - self.desc_wt_time_steps}ts.hdf5",
                        keyz=['coords', 'desc_wts'],
                        valz=[self._parent, self._desc_wts])
                if self._deb_desc_wt_tracker:
                    fname = f"{self.output_folder}/wfns/{self.sim_name}_desc_wt_tracker_{prop_step + 1 - self.desc_wt_time_steps}ts.npy"
                    np.save(fname, np.array(desc_wt_history))

            # Explicitly flush log file every 10 time steps, since it gets caught in buffer in HPC systems
            if prop_step % 10 == 0:
                self._logger.fl.flush()

    def run(self):
        """This function calls propagate and saves simulation results"""
        print("Starting Simulation...")
        dmc_time_start = time.time()
        try:
            throw_error = None
            self.propagate()
            # Delete all checkpoints, since this is the end of the run
            FileManager.delete_older_checkpoints(self.output_folder,
                                                 self.sim_name,
                                                 self.cur_timestep)
        except Exception as e:
            import traceback
            print("ERROR! An error occurred while running the DMC simulation. Dumping a final checkpoint...")
            print("Ignore Approximate ZPE!!!")
            traceback.print_exc()
            throw_error = e
        finally:
            self._logger.final_chkpt()
            self._logger = None
            SimArchivist.chkpt(self, self.cur_timestep)

            # Convert vref vs tau to wavenumbers
            _vref_wvn = Constants.convert(self._vref_vs_tau, "wavenumbers", to_AU=False)

            print("Simulation Complete")
            print('Approximate ZPE', np.average(_vref_wvn[len(_vref_wvn) // 4:]))
            if self.impsamp_manager is not None:
                """Replace discrete integers with the effective time step put forth by the metropolis criteria"""
                ts = self.eff_ts
            else:
                ts = np.arange(len(_vref_wvn)) * self.delta_t

            # Save siminfo
            SimArchivist.save_h5(fname=f"{self.output_folder}/{self.sim_name}_sim_info.hdf5",
                                 keyz=['vref_vs_tau', 'pop_vs_tau', 'atomic_nums', 'atomic_masses'],
                                 valz=[np.column_stack((ts, self._vref_vs_tau)),
                                       np.column_stack((ts, self._pop_vs_tau)),
                                       self._atm_nums, self.masses])

            if self.adiabatic_dmc is not None:
                np.save(f"{self.output_folder}/{self.sim_name}_lambda.npy", self.ad_lam_array)

            finish = time.time() - dmc_time_start

        self._logger = SimLogger(f"{self.output_folder}/{self.sim_name}_log.txt")
        self._logger.finish_sim(finish)
        if throw_error is not None:
            raise throw_error

    def __deepcopy__(self, memodict={}):
        """
        Helper internal copying system for checkpointing.
        Here to ensure that the potential is not pickled.
        """
        cls = self.__class__
        res = cls.__new__(cls)
        memodict[id(self)] = res
        no_gos = ['potential', 'potential_info', 'impsamp_manager', 'impsamp', 'imp_info','adiabatic_dmc','ad_obs_func']
        for k, v in self.__dict__.items():
            if k not in no_gos:
                setattr(res, k, copy.deepcopy(v, memodict))
        return res


def dmc_restart(potential, chkpt_folder, sim_name, additional_timesteps=0, impsamp=None):
    """TODO: Need to add in impsamp infrastructure here for restarting..."""
    dmc_sim = SimArchivist.reload_sim(chkpt_folder, sim_name)
    # Update simulation parameters based on additional timesteps
    dmc_sim._init_restart(additional_timesteps, impsamp)
    # Re-initialize the potential and the logger, as those are not pickleable
    dmc_sim.potential = potential.getpot
    dmc_sim.potential_info = vars(dmc_sim.potential)
    dmc_sim._logger = SimLogger(f"{dmc_sim.output_folder}/{dmc_sim.sim_name}_log.txt")
    # Delete future checkpoints. This shouldn't do anything at this stage, but just to be safe
    FileManager.delete_future_checkpoints(chkpt_folder, sim_name, dmc_sim.cur_timestep)
    return dmc_sim


if __name__ == "__main__":
    print('Hi, this is not how to run this code. Refer to documentation.')
