from .Constants import *


class SimLogger:
    """A utility class for logging the simulation, writes to .log file."""

    def __init__(self, fname, overwrite=False):
        if overwrite:
            self.fl = open(fname, 'w')
        else:
            self.fl = open(fname, 'a')

    def finish_sim(self):
        self.fl.write("Simulation has finished.\n\n")
        self.fl.close()

    def write_ts(self, cur_time_step):
        self.fl.write(f"Time step {cur_time_step}\n")

    def write_chkpt(self, cur_time_step):
        self.fl.write(f"Checkpointing, time step {cur_time_step}\n")

    def write_wfn_save(self, cur_time_step):
        self.fl.write(f"Starting descendant weighting, time step {cur_time_step}\n")
        self.fl.write(f"Will save wave function from time step {cur_time_step}\n")

    def write_desc_wt(self, cur_time_step):
        self.fl.write(f"Finished descendant weighting, time step {cur_time_step}\n")
        self.fl.write(f"Saving Wave function with descendant weights, time step {cur_time_step}\n")

    def write_pot_time(self, cur_time_step, pot_time, maxpot, minpot, avgpot):
        maxpot = Constants.convert(maxpot, 'wavenumbers', to_AU=False)
        minpot = Constants.convert(minpot, 'wavenumbers', to_AU=False)
        avgpot = Constants.convert(avgpot, 'wavenumbers', to_AU=False)
        self.fl.write(f"Potential call time at time step {cur_time_step}:\n")
        self.fl.write(f"\t{pot_time} seconds\n")
        self.fl.write(f"\tAverage energy of ensemble: {avgpot} wavenumbers (without vref correction)\n")
        self.fl.write(f"\tHighest energy walker: {maxpot} wavenumbers\n")
        self.fl.write(f"\tLowest energy walker: {minpot} wavenumbers\n")

    def write_branching(self, cur_time_step, weighting, birthdeath_branch):
        if weighting == 'discrete':
            self.fl.write(f"Birth/Death at time step {cur_time_step}:\n")
            self.fl.write(f"\tWalker Births: {birthdeath_branch[0]}\n")
            self.fl.write(f"\tWalker Deaths: {birthdeath_branch[1]}\n")

        elif weighting == 'continuous':
            self.fl.write(f"Branching at time step {cur_time_step}:\n")
            self.fl.write(f"\tWalkers Branched: {birthdeath_branch[0]}:\n")
            self.fl.write(f"\tMax Wt before Branched: {birthdeath_branch[1]}:\n")
            self.fl.write(f"\tMin Wt before Branched: {birthdeath_branch[2]}:\n")

    def write_beginning(self,attribs):
        self.fl.write(f"Simulation {attribs['sim_name']} starting at step {attribs['cur_timestep']}\n")
        self.fl.write(f"Num Walkers: {attribs['num_walkers']}\n")
        self.fl.write(f"Num Time Steps: {attribs['num_timesteps']}\n")
        self.fl.write(f"Weighting Type: {attribs['weighting']}\n")
        self.fl.write(f"Branch every {attribs['branch_every']} Time Step(s)\n")
        self.fl.write(f"Delta Tau: {attribs['delta_t']} a.u.\n")
        self.fl.write(f"Start Structure Array Shape: {attribs['start_structures'].shape}\n")
        self.fl.write(f"Masses of Each Atom: {attribs['masses']}\n")
        self.fl.write(f"Equilibration Steps Before Collecting Wave Functions: {attribs['equil_steps']}\n")
        self.fl.write(f"Checkpoint Every {attribs['chkpt_every']} time steps\n")
        self.fl.write(f"Collect Wave Functions Every {attribs['wfn_every']} Time Steps After Equilibration\n")
        self.fl.write("\n")
