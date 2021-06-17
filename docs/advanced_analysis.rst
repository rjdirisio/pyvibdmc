Advanced Features in PyVibDMC
=========================================================

``PyVibDMC`` has some advanced features that are not usually necessary for running a 'bare bones' DMC simulation.
This is the documentation page that goes over some of them.

Viewing walkers using Jmol/Avogadro
-------------------------------------------------
``PyVibDMC`` has a numpy array <--> .xyz converter built in. In order to write the walkers from the DMC simulation
to file::

    import pyvibdmc as pv
    import numpy as np

    tutorial_sim = pv.SimInfo('pyvibdmc/pyvibdmc/sample_sim_data/tutorial_water_0_sim_info.hdf5')
    increment = 1000
    cds, dws = tutorial_sim.get_wfns(np.arange(2500,9500+increment,increment))
    pv.XYZNPY.write_xyz(coords=cds, fname='water_geoms.xyz', atm_strings=['H','H','O']) #writes the wave functions to file

Once you have the .xyz file, you may then open it in your molecular visualization software of choice. One can imagine
even sorting your top 100 walkers by their descendant weight and writing those to file as well::

    ...
    cds, dws = tutorial_sim.get_wfns(np.arange(2500,9500+increment,increment))
    idx = np.flip(np.argsort(dws)) # sorted array index from highest to lowest descendant weight
    written_cds = cds[idx[:100]]  # index over coordinates, grabbing top 100 from idx[:100]
    pv.XYZNPY.write_xyz(coords=written_cds, fname='top_water_geoms.xyz', atm_strings=['H','H','O']) #writes the wave functions to file


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

    import pyvibdmc as pv

    import numpy as np

    tutorial_sim = pv.SimInfo('pyvibdmc/pyvibdmc/sample_sim_data/tutorial_water_0_sim_info.hdf5')
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
    from pyvibdmc import Constants
    # or just import pyvibdmc as pv and do pv.Constants
    atoms = ["H", "D", "T", "N", "Br"]
    atomic_masses = [pv.Constants.mass(atom) for atom in atoms] # returns in atomic units
    atomic_masses = [pv.Constants.mass(atom, to_AU=False) for atom in atoms] # returns in amu
    one_mass = Constants.mass("O")

If one had a starting structure in angstroms but needed to convert it to Bohr as an input structure, one could go about
it with or without using the Constants module::

    import numpy as np
    from pyvibdmc import Constants

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
    start_structure = pv.Constants.convert(start_structure,'angstroms',to_AU=True)
    # to convert from bohr to angstrom:
    # start_structure = pv.Constants.convert(start_structure,'angstroms',to_AU=False)

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
    harm_coord[0,0,0] = pv.Constants.convert(0.9578400,'angstroms',to_AU=True) # using the Constants class from above!

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
        eq = pv.Constants.convert(eq,'angstroms',to_AU=True) #convert eq structure to bohr
        geoms = np.tile(eq, (len(cds), 1, 1)) #make n copies of start structure, now geoms is a (n, 3, 3) array
        geoms[:,0,0] = cds.squeeze() #put displaced 1D walkers from DMC into the eq structure, just modifying the x part of H
        v = calc_hoh_pot(geoms, len(geoms)) #call potential with full geometry, only the OH stretch is displaced
        return v

Now, we can run the 1D DMC simulation where are walkers are functionally just 1D particles, but the potential is acting
as if it is a full dimensional system.  Of course, the wave functions then will be only 1D in this case::

    import pyvibdmc as pv
    from pyvibdmc import potential_manager as pm

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
    red_coord[0,0,0] = pv.Constants.convert(0.9578400,'angstroms',to_AU=True) #we only need one geometry, PyVibDMC will duplicate it for us.

    # reduced mass - automated way
    mass = pv.Constants.reduced_mass("O-H")

    for sim_num in range(5):
        red_DMC = pv.DMC_Sim(sim_name=f"water1d_dt10_{sim_num}",
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

Calculating Dihedral Angles
-------------------------------------------------
While this is not an advanced quantity to calculate, its usage requires some finesse. The ``AnalyzeWfn.dihedral()`` function
handles both proper and improper dihedral angles the same way.  The equations used to calculate the angles can be found
in this `old wikipedia article <https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors>`_
which cites `this paper <https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1096-987X(19960715)17:9%3C1132::AID-JCC5%3E3.0.CO;2-T>`_

Knowledge of these articles is not necessary to calculate the dihedral angle. All one needs to do is: ::

    import pyvibdmc as pv
    import numpy as np
    analyzer = pv.AnalyzeWfn(coords)
    angle = np.degrees(analyzer_dim.dihedral(atm_1, atm_2, atm_3, atm_4))

All angles in ``PyVibDMC`` are returned in radians, so we can convert to degrees if desired.  The atom numbering is crucial:
For proper dihedral angles (like the carbon chain in butane), simply go from one end of the carbon chain to another (C1-C2-C3-C4).
For improper dihedral angles, such as formaledhyde, you should label the atoms according to H-C-O-H, much like one would
if using GaussView or Avogadro using the "measure" tool.

Performing 3D Rotations of molecules using PyVibDMC
-------------------------------------------------
One can perform 3D Rotations using the AnalyzeWfn tool in ``PyVibDMC``. The way to do this is to use the ``MolRotator``
object, which can generate rotation matrices, rotate molecules and vectors, and generate and extract Euler angles (not rigorously tested).

For rotating a 3 atoms in a molecule to the xy plane, one can use the ``rotate_to_xy_plane`` method::

    from pyvibdmc import MolRotator as rot
    coords = ... #some nxmx3 numpy array or mx3 numpy array
    rot.rotate_to_xy_plane(coords, origin_atm, x_ax_atm, xyp_atm)

Where ``origin_atm`` is the atom index corresponding to the atom that will end up on the origin, the ``x_ax_atm`` on the
x-axis, and the ``xyp_atm`` on the xy-plane.

If you have a bunch of vectors, like the dipole moments for each of your walkers in a DMC simulation, one can rotate
those vectors according to a particular rotation matrix. For example, say I have a rotation matrix for each walker
generated from an Eckart rotation. You can apply the rotation matrix to both the molecule itself but also the dipole
vectors (dipole shape: num_walkersx3)::

    from pyvibdmc import MolRotator as rot
    rot_mats = ... # my num_walkers x 3 x 3 rotation matrices
    vecs = ... # my num_walkers x 3 vectors
    coords = ... # my num_walkers x num_atoms x 3 array

    # Let's rotate each of our dipole vectors according to the corresponding rotation matrix
    rotated_vecs = rot.rotate_vec(rot_mats,vec)
    # Let's rotate each of our walkers according to the corresponding rotation matrix
    rotated_coords = rot.rotate_geoms(rot_mats,coords)

Of course, if you want to apply the same rotation matrix to every walker, you can still use ``rot.rotate_geoms`` but
just make a ``num_walkers x 3 x 3`` copy of your ``3 x 3`` rotation matrix using ``np.tile`` or something similar.

Generating and Extracting Euler Angles
-------------------------------------------------

WARNING: This is not well tested. There may be some phase issues (+/-) in the calculated angles.

The Euler angles that are calculated and extracted in this code use a ``ZYZ`` rotation formalism.  This code is not well
tested and someone may improve upon it in the future.  To generate Euler angles, one needs two coordinate systems.
The ``gen_eulers`` method generates Euler angles that rotate ``xyz`` to ``XYZ``::

    from pyvibdmc import MolRotator as rot
    xyz = ... #For each walker, the coordinate system that will be rotated to the new one. (num_walkers x 3) , or just (3)
    XYZ = ... #For each walker, the coordinate system that xyz will be rotated to (num_walkers x 3), or just (3)
    theta, phi, chi = rot.gen_eulers(xyz,XYZ)

Where ``theta`` is defined from ``0 to pi`` and ``phi`` and ``chi`` are defined from ``0 to 2*pi``.

In order to extract the Euler angles from a rotation matrix, you can use ``extract_eulers``::

    from pyvibdmc import MolRotator as rot
    rot_mats = ... # num_walkers x 3 x 3 rotation matrix, or just 3 x 3
    theta, phi, chi = rot.extract_eulers(rot_mats)

