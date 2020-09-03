"""
pyvibdmc.py
A general purpose diffusion monte carlo code for studying vibrational problems.

This is the main file, which runs the DMC code itself.  To see the basic algorithm in
action, go to self.propagate()

"""
import numpy as np
import matplotlib.pyplot as plt

from .simulation_utilities import *


class DMC_Sim:
    def __init__(self,
                 sim_name="DMC_Sim",
                 output_folder="exSimResults",
                 weighting='discrete',
                 num_walkers=10000,
                 num_timesteps=10000,
                 equil_steps=2000,
                 chkpt_every=1000,
                 wfn_every=1000,
                 desc_steps=50,
                 atoms=[],
                 dimensions=3,
                 delta_t=5,
                 potential=None,
                 masses=None,
                 start_structures=None,
                 branch_every=1,
                 cur_timestep=0
                 ):
        """
        The main object that you will use to run the DMC simulation. You will instantiate an object DMC_SIM(...)
        and then use object.run() to begin the simulation.
        @param sim_name:Simulation name for saving wavefunctions
        @type sim_name:str
        @param output_folder:The folder where the results will be stored, including wavefunctions and energies
        @type output_folder:str
        @param weighting:Discrete or Continuous weighting DMC.  Continuous means that there are fixed number of walkers
        @type weighting:str
        @param num_walkers:Number of walkers we will start the simulation with
        @type num_walkers:int
        @param num_timesteps:Total time steps we will be prop_stepagating the walkers.  num_timesteps*delta_t = total time in A.U.
        @type num_timesteps:int
        @param equil_steps: Time before we start collecting wavefunctions
        @type equil_steps:int
        @param chkpt_every:How many time steps in between we will propagate before collecting another wavefunction
        every time we collect a wavefunction, we checkpoint the simulation
        @type chkpt_every:int
        @param desc_steps:Number of time steps for descendant weighting.
        @type desc_steps: int
        @param atoms:List of atoms for the simulation
        @type atoms:list
        @param dimensions: 3 leads to a 3N dimensional simulation. This should always be 3 for real systems.
        @type dimensions:int
        @param delta_t: The length of the time step; how many atomic units of time are you going in one time step.
        @type delta_t: int
        @param potential: Takes in coordinates, gives back energies
        @type potential: function
        @param masses:For feeding in artificial masses in atomic units.  If not, then the atoms param will designate masses
        @type masses: list
        @param start_structures:An initial structure to initialize all your walkers
        @type start_structures:np.ndarray
        @param branch_every:Branch the the walkers (cont. weighting, check if weights are too high, discr. weighting, if
                                                    the walker's potential value is too high).
        @type branch_every:int
        @param cur_timestep: The current time step, should be zero unless you are restarting
        @type cur_timestep: int0
        """

        self.atoms = atoms
        self.sim_name = sim_name
        self.output_folder = output_folder
        self.num_walkers = num_walkers
        self.num_timesteps = num_timesteps
        self.potential = potential
        self.weighting = weighting.lower()
        self.desc_steps = desc_steps
        self.branch_every = branch_every
        self.delta_t = delta_t
        self.start_structures = start_structures
        self.masses = masses
        self.equil_steps = equil_steps
        self.chkpt_every = chkpt_every
        self.wfn_every = wfn_every
        self.dimensions = dimensions
        self.cur_timestep = cur_timestep
        self._initialize()

    def _initialize(self):
        """
        Initialization of all the arrays and internal simulation parameters.
        """
        # Initialize the rest of the (private) variables needed for the simulation
        # Arrays used to mark important events througout the simulation
        self._prop_steps = np.arange(self.cur_timestep, self.num_timesteps)
        self._branch_step = np.arange(self.cur_timestep, self.num_timesteps + self.branch_every, self.branch_every)
        self._chkptStep = np.arange(self.equil_steps, self.num_timesteps + self.chkpt_every, self.chkpt_every)
        self._wfnSaveStep = np.arange(self.equil_steps, self.num_timesteps + self.wfn_every, self.wfn_every)
        self._dwSaveStep = self._wfnSaveStep + self.desc_steps

        # Arrays that carry data throughout the simulation
        self._who_from = None  # Descendant weighting doesn't happen right away, no need to init
        self._walker_pots = np.zeros(self.num_walkers)
        self._vref_vs_tau = np.zeros(self.num_timesteps)
        self._pop_vs_tau = np.zeros(self.num_timesteps)

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
            raise Exception("You broke the initial walkers input or didn't provide a starting geometry..."
                            "The format is (nxmxd), where n = 1 or num_walkers, m = num_atoms, d = dims")
        if self.masses is None:
            self.masses = np.array([Constants.mass(a) for a in self.atoms])
        if len(self.masses) != len(self.atoms):
            raise Exception("Your number of atoms list does not match your number of self.masses you provided.")
        self._atm_nums = get_atomic_num(self.atoms)

        # Constants for simulation
        self._sigmas = np.sqrt((self.delta_t) / self.masses)
        self._alpha = 1.0 / (2.0 * self.delta_t)

        # Where to save the data
        fileManager.create_filesystem(self.output_folder)

        #Weighting technique
        if self.weighting == 'continuous':
            self.contWts = np.ones(self.num_walkers)
        else:
            self.contWts = None


    def birthOrDeath_vec(self, Desc):
        """
        Chooses whether or not the walker made a bad enough random walk to be removed from the simulation.
        For discrete weighting, this leads to removal or duplication of the walkers.  For continuous, this leads
        to an update of the weights and a potential branching of a large weight walker to the smallest one
        @param Desc: A boolean that checks if descendent weighting should be occuring right now.
        @return: Updated Continus Weights , the "who from" array for descendent weighting, walker coords, and pot vals.
         """
        if self.weighting == 'discrete':
            randNums = np.random.random(len(self._walker_coords))

            deathMask = np.logical_or((1 - np.exp(-1. * (self._walker_pots - self._vref) * self.delta_t)) < randNums,
                                      self._walker_pots < self._vref)
            self._walker_coords = self._walker_coords[deathMask]
            self._walker_pots = self._walker_pots[deathMask]
            randNums = randNums[deathMask]
            if Desc:
                self._who_from = self._who_from[deathMask]

            exTerm = np.exp(-1. * (self._walker_pots - self._vref) * self.delta_t) - 1
            ct = 1
            while np.amax(exTerm) > 0.0:
                randNums = np.random.random(len(self._walker_coords))
                birthMask = np.logical_and(exTerm > randNums,
                                           self._walker_pots < self._vref)
                self._walker_coords = np.concatenate((self._walker_coords,
                                                      self._walker_coords[birthMask]))
                self._walker_pots = np.concatenate((self._walker_pots,
                                                    self._walker_pots[birthMask]))
                if Desc:
                    self._who_from = np.concatenate((self._who_from,
                                                     self._who_from[birthMask]))
                exTerm = np.exp(-1. * (self._walker_pots - self._vref) * self.delta_t) - (1+ct)
                ct += 1


        else:
            self.contWts = self.contWts * np.exp(-1.0 * (self._walker_pots - self._vref) * self.delta_t)
            thresh = 1.0 / self.num_walkers
            killMark = np.where(self.contWts < thresh)[0]
            for walker in killMark:
                maxWalker = np.argmax(self.contWts)
                self._walker_coords[walker] = np.copy(self._walker_coords[maxWalker])
                self._walker_pots[walker] = np.copy(self._walker_pots[maxWalker])
                if Desc:
                    self._who_from[walker] = self._who_from[maxWalker]
                self.contWts[maxWalker] /= 2.0
                self.contWts[walker] = np.copy(self.contWts[maxWalker])
        return self.contWts, self._who_from, self._walker_coords, self._walker_pots

    def moveRandomly(self, _walker_coords):
        """
        The random displacement of each of the coordinates of each of the walkers, done in a vectorized fashion
        @param _walker_coords: The coordinates array that stores the xyz vals of each of the walkers.
        @return: The displaced walker coordinates
        """
        disps = np.random.normal(0.0,
                                 self._sigmas,
                                 size=np.shape(_walker_coords.transpose(0, 2, 1))).transpose(0, 2, 1)
        return _walker_coords + disps

    def getVref(self):  # Use potential of all walkers to calculate self._vref
        """
             Use the energy of all walkers to calculate self._vref with a correction for the fluctuation in the population
             or weight.
         """
        if self.weighting == 'discrete':
            Vbar = np.average(self._walker_pots)
            correction = (len(self._walker_pots) - self.num_walkers) / self.num_walkers
        else:
            Vbar = np.average(self._walker_pots, weights=self.contWts)
            correction = (np.sum(self.contWts - np.ones(self.num_walkers))) / self.num_walkers
        self._vref = Vbar - (self._alpha * correction)

    def countUpDescWeights(self, dwts):
        """
        At the end of descendent weighting, count up which walkers came from other walkers (descendants)
        @param dwts: The array in which the descendent weights will be stored
        """
        if self.weighting == 'discrete':
            unique, counts = np.unique(self._who_from, return_counts=True)
            dwts[unique] = counts
        else:
            for q in range(len(self.contWts)):
                dwts[q] = np.sum(self.contWts[self._who_from == q])

    def updateSimArs(self, prop_step, Vref):
        """
        @param prop_step:  the time step that we are currently on
        @param Vref: the reference energy that will be added to the internal lists.
        """
        self._vref_vs_tau[prop_step] = Vref
        if self.weighting == 'discrete':
            self._pop_vs_tau[prop_step] = len(self._walker_coords)
        else:
            self._pop_vs_tau[prop_step] = np.sum(self.contWts)

    def propagate(self):
        """
             The main DMC loop.
             1. Move Randomly
             2. Calculate the Potential Energy
             3. Birth/Death
             4. Update Vref
             Additionally, checks when the wavefunction has hit a point where it should save / start descendent
             weighting.
         """
        DW = False
        for prop_step in self._prop_steps:
            self.cur_timestep = prop_step
            if self.cur_timestep % 10 == 0:
                print(self.cur_timestep)
            self._walker_coords = self.moveRandomly(self._walker_coords)
            self._walker_pots = self.potential(self._walker_coords)

            if prop_step == self._prop_steps[0]:
                self.getVref()

            if prop_step in self._chkptStep:
                SimArchivist.chkpt(self, prop_step)

            if prop_step in self._wfnSaveStep:
                dwts = np.zeros(len(self._walker_coords))
                parent = np.copy(self._walker_coords)
                self._who_from = np.arange(len(self._walker_coords))
                DW = True

            if prop_step in self._dwSaveStep:
                DW = False
                self.countUpDescWeights(dwts=dwts)
                vref_cm = Constants.convert(self._vref_vs_tau, 'wavenumbers', to_AU=False)
                SimArchivist.saveH5(
                    fname=f"{self.output_folder}/wfns/{self.sim_name}_wfn_{prop_step - self.desc_steps}ts.hdf5",
                    keyz=['coords', 'desc_weights', 'desc_time', 'atoms', 'vref'],
                    valz=[parent,
                          dwts,
                          self.desc_steps,
                          self._atm_nums,
                          vref_cm])

            if prop_step in self._branch_step:
                self.contWts, self._who_from, self._walker_coords, self._walker_pots = self.birthOrDeath_vec(DW)
            else:
                if self.weighting == 'continuous':
                    self.contWts = self.contWts * np.exp(-1.0 * (self._walker_pots - self._vref) * self.delta_t)

            self.getVref()
            self.updateSimArs(prop_step, self._vref)

    def run(self):
        """This function calls propagate and plots vref vs tau"""
        self.propagate()
        _vrefCM = Constants.convert(self._vref_vs_tau, "wavenumbers", to_AU=False)
        print('approximate ZPE (temporary printout)', np.average(_vrefCM[100:]))

        plt.plot(_vrefCM)
        plt.xlabel("Time Step")
        plt.ylabel("Vref (wavenumbers)")
        plt.savefig(f"{self.output_folder}/vrefVsTau.png")
        plt.close()
        ts = np.arange(len(_vrefCM))
        SimArchivist.saveH5(fname=f"{self.output_folder}/{self.sim_name}_simInfo.hdf5",
                            keyz=['vrefVsTau', 'popVsTau'],
                            valz=[np.column_stack((ts, _vrefCM)), np.column_stack((ts, self._pop_vs_tau))])

    def __deepcopy__(self, memodict={}):
        """Helper internal copying system for checkpointing."""
        cls = self.__class__
        res = cls.__new__(cls)
        memodict[id(self)] = res
        for k,v in self.__dict__.items():
            if k != 'potential':
                setattr(res,k,copy.deepcopy(v,memodict))
        return res

def DMC_Restart(potential,
                time_step,
                chkpt_folder="exSimulation_results/",
                sim_name='DMC_Sim'):
    dmc_sim = SimArchivist.reloadSim(potential,
                                     chkpt_folder,
                                     sim_name,
                                     time_step)
    fileManager.delete_future_checkpoints(chkpt_folder, sim_name, time_step)
    return dmc_sim

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print('hi')
