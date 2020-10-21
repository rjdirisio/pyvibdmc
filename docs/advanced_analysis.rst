Advanced Features in PyVibDMC
=========================================================

``PyVibDMC`` has some advanced features that are not usually necessary for running a 'bare bones' DMC simulation.
This is the documentation page that goes over some of them.

Viewing walkers using Jmol/Avogadro
-------------------------------------------------
``PyVibDMC`` has a numpy array <--> .xyz converter built in. In order to write the walkers from the DMC simulation
to file::

    from pyvibdmc.analysis import * # this imports AnalyzeWfn as well as Plotter
    from pyvibdmc.simulation_utilities import xyz_npy as xyz

    import numpy as np

    tutorial_sim = SimInfo('pyvibdmc/pyvibdmc/sample_sim_data/tutorial_water_0_sim_info.hdf5')
    increment = 1000
    cds, dws = tutorial_sim.get_wfns(np.arange(2500,9500+increment,increment))
    xyz.write_xyz(coords=cds, fname='water_geoms.xyz', atm_strings=['H','H','O']) #writes the wave functions to file

Once you have the .xyz file, you may then open it in your molecular visualization software of choice. One can imagine
even sorting your top 100 walkers by their descendant weight and writing those to file as well::

    ...
    cds, dws = tutorial_sim.get_wfns(np.arange(2500,9500+increment,increment))
    idx = np.flip(np.argsort(dws)) # sorted array index from highest to lowest descendant weight
    written_cds = cds[idx[:100]]  # index over coordinates, grabbing top 100 from idx[:100]
    xyz.write_xyz(coords=written_cds, fname='top_water_geoms.xyz', atm_strings=['H','H','O']) #writes the wave functions to file


Working with HDF5 files in Python (and how not to)
---------------------------------------------------
``PyVibDMC`` outputs .hdf5 files that can be easily read in Python using H5Py.  This section will
detail the contents of the .hdf5 files and how to load them in manually.  It will also show you how to access all the
information in the hdf5 files without knowing how H5Py works at all.

H5Py allows reading .hdf5 files as if they are dictionaries, and allow us to use normal NumPy syntax.

The ``*sim_info.hdf5`` files produced by PyVibDMC have numerous arrays stored in them.  They are keyed by:

- 'vref_vs_tau'
- 'pop_vs_tau'
- 'atomic_nums'
- 'atomic_masses'

The names should be quite self explanatory. The first is the vref array, which stores the energies at each time step,
the next stores the population (either the size of the ensemble for discrete weighting or the sum of the
weights for continuous weighting), and the third tells about the ordering of the atoms in the simulation; if you were
running a water simulation you could run it doing [1,1,8], [8,1,1], etc, the fourth returns the each atom's masses.
If you were running a DOD calculation, you would get [1,8,1] and [...], respectively.

To access these arrays manually, you can use similar code from ``extract_sim_info.py``::

    import numpy as np
    import h5py

    with h5py.File(fname, 'r') as f:
        vref_vs_tau = f['vref_vs_tau'][:] # (num_timesteps,2) numpy array (time step, energies)
        pop_vs_tau = f['pop_vs_tau'][:] # (num_timesteps,2) numpy array (time step, population)
        atom_nums = f['atomic_nums'][:] # list of ints
        atom_masses = f['atomic_masses'][:] #list of floats

The ``wfns/*ts.hdf5`` files contain the walkers and the descendant weights, was performed. This array would be manually
accessed in the same way::

    import numpy as np
    import h5py

    with h5py.File(fname, 'r') as f:
        cds = f['coords'][:] # (n,m,3) numpy array
        dwts = f['desc_weights'][:] # (n) numpy array

However, ``PyVibDMC`` has ways to extract these arrays so that the user does not even need to know how to manipulate .hdf5
files::

    from pyvibdmc.analysis import * # this imports SimInfo, AnalyzeWfn as well as Plotter
    from pyvibdmc.simulation_utilities import xyz_npy as xyz

    import numpy as np

    tutorial_sim = SimInfo('pyvibdmc/pyvibdmc/sample_sim_data/tutorial_water_0_sim_info.hdf5')
    vref_vs_tau =  tutorial_sim.get_vref()
    pop_vs_tau =  tutorial_sim.get_pop()
    atom_nums  = tutorial_sim.get_atomic_nums()
    atom_masses = tutorial_sim.get_atom_masses()

    #of course, from the tutorial, one can use .get_wfns() to get cds, dws

Advanced and Debug Keyword Arguments in DMC_Sim
-------------------------------------------------------

- DEBUG_alpha: The number that will be used instead of 1/(2*delta_t) for alpha. This number modulates the fluctuation of
  vref throughout the simulation. A smaller alpha value leads to a more stringent fluctuation. However, as alpha
  increaese, the previous time step's vref has more impact on the current time step's vref. This is not proper for a
  stochastic simulation and should be avoided. There should be a strong fluctuation of vref about the zero-point energy
  for almost the entire simulation.

- DEBUG_save_desc_wt_tracker: This boolean argument will save the "who_from" array. During descendant weighting, one
  needs to keep track of which new walkers come from branching. In order to accomplish this, a "who_from" array is
  initalized at the beginning of the descendant weighting cycle, and is used at the end to count up descendants. If one
  would like this array at every time step within the descendant weighting cycle to be saved, you can turn this on.
  The only reason you would want to do this is so that you can calculate the descendant weights at each time step during
  the cycle, i.e. if you were trying to understand how Psi^2 looks different depending on how long you descendant weight
  for.

- DEBUG_save_training_every: This is a variable that saves the potential energies and coordinates of each walker
  in the ensemble every x time steps.

- branch_every: This argument will not branch (do births and deaths) at every step of the DMC simulation.  This is
  typically for high-performance computing environments to eliminate cross-node communication. HPC DMC is not currently
  implemented, so this argument should always be 1.

Performing 3D Rotations of atoms using PyVibDMC
-------------------------------------------------
Documentation pending.
