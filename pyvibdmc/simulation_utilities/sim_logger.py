from .Constants import *


class SimLogger:
    """A utility class for logging the simulation, writes to .log file."""

    def __init__(self, fname, overwrite=False):
        if overwrite:
            self.fl = open(fname, 'w')
        else:
            self.fl = open(fname, 'a')

    def finish_sim(self):
        self.fl.write("Simulation has finished.\n")
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
