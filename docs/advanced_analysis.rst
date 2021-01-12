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

- ``'vref_vs_tau'``
- ``'pop_vs_tau'``
- ``'atomic_nums'``
- ``'atomic_masses'``

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

- ``DEBUG_alpha``: The number that will be used instead of 1/(2*delta_t) for alpha. This number modulates the fluctuation of
  vref throughout the simulation. A smaller alpha value leads to a more stringent fluctuation. However, as alpha
  increaese, the previous time step's vref has more impact on the current time step's vref. This is not proper for a
  stochastic simulation and should be avoided. There should be a strong fluctuation of vref about the zero-point energy
  for almost the entire simulation.

- ``DEBUG_save_desc_wt_tracker``: This boolean argument will save the "who_from" array. During descendant weighting, one
  needs to keep track of which new walkers come from branching. In order to accomplish this, a "who_from" array is
  initalized at the beginning of the descendant weighting cycle, and is used at the end to count up descendants. If one
  would like this array at every time step within the descendant weighting cycle to be saved, you can turn this on.
  The only reason you would want to do this is so that you can calculate the descendant weights at each time step during
  the cycle, i.e. if you were trying to understand how Psi^2 looks different depending on how long you descendant weight
  for.

- ``DEBUG_save_training_every``: This is a variable that saves the potential energies and coordinates of each walker
  in the ensemble every x time steps.

- ``DEBUG_mass_change``: This changes the mass by a factor of ``factor_per_change`` every ``change_every`` time steps of the
  simulation.  For example, if one starts off the simulation with a massive 50x regular mass atom set, every x time steps
  the mass will go from 50x as massive to 25x to  12.5x ... until the simulation finishes if ``factor_per_change=0.5``.
  You will run into issues if you change the masses by a lot quickly. All your walkers die as they will explore
  farther out in the PES too quickly. You can also feed in an array to ``factor_per_change``, which could make it linear
  scaling instead of logarithmic. For example, if one started off with 5x massive atoms, you can then do something like
  ``factor_array=[1,1,1,1,....,4/5,1,1,1,1....,3/4,1,1,1,1....2/3,...]`` and set ``change_every=1`` to decrease it from
  5x as massive to 4x massive to 3x massive...which will happen every n time steps.

- ``branch_every``: This argument will not branch (do births and deaths) at every step of the DMC simulation.  This is
  typically for high-performance computing environments to eliminate cross-node communication. HPC DMC is not currently
  implemented, so this argument should always be 1.

- ``cont_wt_thresh``: This argument only does anything when you are using continuous weighting.  If this is a single number, it is
  specifying the lower bound on the allowable walker weight in the simulation (if it gets below this number, the walker will
  be removed and the highest weight walker will be split into two walkers at the same coordinate but with 1/2 the weight).
  If it is two numbers, the first number will be the lower bound, and the second number will be ther upper bound (if it
  gets above this number, the walker will be split into two, and the smallest available weight walker will be removed
  from the simulation).


The Constants Module: A Unit Converter and Atom Data Holder
-------------------------------------------------------------
Inside ``PyVibDMC`` there is a (very) limited unit converter and atomic data storage module called ``Constants``.  The first
version of this small class was written by `Mark Boyer <https://github.com/b3m2a1>`_.  This class is completely
optional to use, but some may find it useful in preparing their DMC simulations, and it is used throughout ``PyVibDMC``.

The three unit conversions Constants can do are as follows:

- Bohr <--> Angstroms. ``Constants.convert(nparray_or_float, 'angstroms',to_AU=TrueOrFalse)``

- Hartree <--> Wavenumbers ``Constants.convert(nparray_or_float, 'wavenumbers',to_AU=TrueOrFalse)``

- Mass of Electron <--> amu ``Constants.convert(nparray_or_float, 'amu',to_AU=TrueOrFalse)``

Additionally, Constants houses the masses of the most common isotopes of the atoms on the periodic table (data
from `NIST <https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses>`_),
and also includes the mass of deuterium and tritium::

    import numpy as np
    from pyvibdmc.simulation_utilities import * # imports Constants
    atoms = ["H", "D", "T", "N", "Br"]
    atomic_masses = [Constants.mass(atom) for atom in atoms] # returns in atomic units
    atomic_masses = [Constants.mass(atom, to_AU=False) for atom in atoms] # returns in amu
    one_mass = Constants.mass("O")

If one had a starting structure in angstroms but needed to convert it to Bohr as an input structure, one could go about
it with or without using the Constants module::

    import numpy as np
    from pyvibdmc.simulation_utilities import * # imports Constants

    # Scenario 1: not using Constants
    bohr_to_ang = 0.529177 # multiply something in bohr by this to get to angstroms
    ang_to_bohr = 1/bohr_to_ang
    start_structure = np.array([[0.9578400,0.0000000,0.0000000],
                                [-0.2399535,0.9272970,0.0000000],
                                [0.0000000,0.0000000,0.0000000]])
    start_structure *= ang_to_bohr

    # Scenario 2: Using Constants
    start_structure = np.array([[0.9578400,0.0000000,0.0000000],
                                [-0.2399535,0.9272970,0.0000000],
                                [0.0000000,0.0000000,0.0000000]])
    start_structure = Constants.convert(start_structure,'angstroms',to_AU=True)
    # to convert from bohr to angstrom:
    # start_structure = Constants.convert(start_structure,'angstroms',to_AU=False)

Reduced-Dimensional DMC Calculations: Example
-------------------------------------------------
Say one wanted to run only a DMC simulation on a particular degree of freedom in a particular molecular system. For
example, what if you wanted to run a DMC simulation on *just* one OH stretch in water? To do this, we can play a few
tricks to get it to work in the confines of ``PyVibDMC``.

To begin, we will use the equilibrium structure where one of the two stretching atoms is on the origin,
and the other is on the x-axis in 3D space.  For our example, the oxygen will be at the origin and one of the
hydrogen atoms will be on the x-axis::

    import numpy as np
    start_structure = np.array([[0.9578400,0.0000000,0.0000000],
                                [-0.2399535,0.9272970,0.0000000],
                                [0.0000000,0.0000000,0.0000000]])

However, we will *not* give this structure to ``DMC_Sim``, but will only show it to the
``potential_manager``. More on this later.

We can set up a 1-Dimensional DMC simulation, where we are just propagating the x-component
of the hydrogen we want to move, in this case the coordinate ``start_structure[0,0]``.
So, we will set up a 1D DMC starting structure::

    harm_coord = np.zeros((1,1,1)) # we are going to set up our initial ensemble to be (n, 1, 1) numpy array
    harm_coord[0,0,0] = Constants.convert(0.9578400,'angstroms',to_AU=True) # using the Constants class from above!

Now, we will modify our potential energy call, as the coordinates passed to the potential will be n_walkers x 1 x 1::

    # h2o_potential.py
    from h2o_pot import calc_hoh_pot
    import numpy as np

    # we will not be calling this
    def water_pot(cds):
        return calc_hoh_pot(cds, len(cds))

    #call this!
    def water_pot_1d(cds):
        """Passes in a (n,1,1) array from DMC_Sim"""
        eq = np.array([[0.9578400,0.0000000,0.0000000],
             [-0.2399535,0.9272970,0.0000000],
             [0.0000000,0.0000000,0.0000000]])
        eq = Constants.convert(eq,'angstroms',to_AU=True) #convert eq structure to bohr
        geoms = np.tile(eq, (len(cds), 1, 1)) #make n copies of start structure, now geoms is a (n, 3, 3) array
        geoms[:,0,0] = cds.squeeze() #put displaced 1D walkers from DMC into the eq structure, just modifying the x part of H
        v = calc_hoh_pot(geoms, len(geoms)) #call potential with full geometry, only the OH stretch is displaced
        return v

Now, we can run the 1D DMC simulation where are walkers are functionally just 1D particles, but the potential is acting
as if it is a full dimensional system.  Of course, the wave functions then will be only 1D in this case::

    import pyvibdmc as dmc
    from pyvibdmc import potential_manager as pm
    from pyvibdmc.simulation_utilities import *

    pot_dir = 'Path/To/Partridge_Schwenke_H2O' #this directory is the one you copied that is outside of pyvibdmc.
    py_file = 'h2o_potential.py'
    pot_func = 'water_pot_1d'

    ps_oh = pm.Potential(potential_function=pot_func,
                                   python_file=py_file,
                                   potential_directory=pot_dir,
                                   num_cores=2
                            )

    # Equilibrium "geometry" of the 1d harmonic oscillator in *atomic units*,
    red_coord = np.zeros((1,1,1))
    red_coord[0,0,0] = Constants.convert(0.9578400,'angstroms',to_AU=True) #we only need one geometry, PyVibDMC will duplicate it for us.

    # reduced mass - automated way
    mass = Constants.reduced_mass("O-H")

    for sim_num in range(5):
        red_DMC = dmc.DMC_Sim(sim_name=f"water1d_dt10_{sim_num}",
                               output_folder="red_dim_dmc",
                               weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                               num_walkers=10000, #number of geometries exploring the potential surface
                               num_timesteps=10000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                               equil_steps=1000, #how long before we start collecting wave functions
                               chkpt_every=9800, #checkpoint the simulation every "chkpt_every" time steps
                               wfn_every=5000, #collect a wave function every "wfn_every" time steps
                               desc_wt_steps=50, #number of time steps you allow for descendant weighting per wave function
                               atoms=['X'], #It doesn't matter what atom you put here if using custom mass.
                               delta_t=1, #the size of the time step in atomic units
                               potential=ps_oh,
                               start_structures=red_coord,
                               masses=mass #can put in artificial masses, otherwise it auto-pulls values from the atoms string
        )
        red_DMC.run()

Performing 3D Rotations of atoms using PyVibDMC
-------------------------------------------------
Documentation pending.
