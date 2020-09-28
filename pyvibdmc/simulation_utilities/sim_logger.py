
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

    def write_ts(self,cur_time_step):
        self.fl.write(f"Time step {cur_time_step}\n")

    def write_chkpt(self, cur_time_step):
        self.fl.write(f"Checkpointing, time step {cur_time_step}\n")

    def write_wfn_save(self, cur_time_step):
        self.fl.write(f"Starting descendant weighting, time step {cur_time_step}\n")
        self.fl.write(f"Will save wave function from time step {cur_time_step}\n")

    def write_desc_wt(self, cur_time_step):
        self.fl.write(f"Finished descendant weighting, time step {cur_time_step}\n")
        self.fl.write(f"Saving Wave function with descendant weights, time step {cur_time_step}\n")

    def write_pot_time(self,cur_time_step, pot_time):
        self.fl.write(f"Potential call time at time step {cur_time_step}:\n")
        self.fl.write(f"\t{pot_time} seconds\n")

    def write_branching(self, cur_time_step, weighting, birthdeath_branch):
        if weighting == 'discrete':
            self.fl.write(f"Birth/Death at time step {cur_time_step}:\n")
            self.fl.write(f"Walker Births: {birthdeath_branch[0]}\n")
            self.fl.write(f"Walker Deaths: {birthdeath_branch[1]}\n")

        elif weighting == 'continuous':
            self.fl.write(f"Branching at time step {cur_time_step}:\n")
            self.fl.write(f"Walkers Branched: {birthdeath_branch[0]}:\n")
