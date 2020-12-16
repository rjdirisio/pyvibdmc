Getting Started
===============

This page details how to get started with `PyVibDMC <https://github.com/rjdirisio/pyvibdmc>`_.

The package is hosted on `PyPi <https://pypi.org/project/pyvibdmc/>`_. To install it, please use::

    pip install pyvibdmc

Theory of Diffusion Monte Carlo (DMC)
-------------------------------------------------------
If you do not have experience with Diffusion Monte Carlo, I suggest reading the
`McCoy group's tutorial on and explanation of DMC <https://mccoygroup.github.io/References/References/Monte%20Carlo%20Methods/DMC.html>`_.

This reference also has links to academic publications that detail the method further.

Installation
--------------
PyVibDMC is tested on and is for Mac and Linux architectures.

If using Windows, please use Windows Subsystem for Linux (WSL). However, there are a few known issues with how PyVibDMC
interacts with WSL. These include:

- (WSL v1) After a complete simulation, if using parallelization, the python process may hang until manually terminated. The way to currently circumvent this is through using ``pm.Potential.mp_close()`` at the end of your run script.
- (WSL v2) If a Fortran potential reads a data file during its F2PY function call, it will seg fault or be unable to read the data file.

These issues are not present on Mac or Linux.

Dependencies (All pre-installed with Anaconda3)
-------------------------------------------------------
- numpy
- matplotlib
- h5py
- Tutorial: Compiler for the potential energy surface (the tutorial potential uses gfortran)
- Tutorial: make (on Linux systems, this is usually installed via the 'build-essential' or 'Development Tools' packages)

Extracting Sample Potentials and Sample DMC Results
------------------------------------------------------
The pyvibdmc package has two directories that are useful for the tutorial: ``sample_potentials`` and ``sample_sim_data``.
To access these files, you will need to ``cp`` them out of the ``PyVibDMC`` package. To locate the package, first do::

    pip show pyvibdmc

This should tell you the ``Location`` of the installation. An example installation location is in
``/home/<username>/.local/lib/python3.8/site-packages``.  To copy the sample directories out, then use ``cp``::

    cp -r <path_to_installation>/pyvibdmc/pyvibdmc/sample_potentials/ .
    cp -r <path_to_installation>/pyvibdmc/pyvibdmc/sample_sim_data/ .

Initial Conditions in The DMC
---------------------------------------------
Traditionally, the starting structure that is used at the beginning of the DMC simulation is the equilibrium structure
for the potential energy surface that is being used.  The structure is then "blown up" by 1.01 to help the walkers
sample farther out in the potential more quickly.  You will see an example of this in the tutotial run script down below.

However, there are more sophisticated methods to generate initial conditions.  For example, one can perform a harmonic
frequency calculation, generate Cartesian normal modes, and then sample from the harmonic ground state wave function
before the beginning of the DMC. One can also permute like atoms before running a simulation.  To do these types of things,
please look at the `Initial Conditions page <https://pyvibdmc.readthedocs.io/en/latest/initial_conditions.html>`_.


Tutorial: Water Monomer
------------------------
Once ``pip`` installed, one can ``import pyvibdmc`` from any directory.

Before running the simulation,
please read about
how `PyVibDMC handles external potential energy surfaces <https://pyvibdmc.readthedocs.io/en/latest/potentials.html>`_

This example script runs 5 DMC simulations on a single water molecule (H\ :sub:`2`\ O)
using the Fortran Potential Energy Surface built by Partridge and Schwenke.  This potential energy surface is located
in the PyVibDMC package, at ``sample_potentials/FortPots/Partridge_Schwenke_H2O``. To expose the Fortran subroutine to Python,
please ``cd`` into the directory you copied, and run ``make``. This will build a ``.so`` file that is called
in ``h2o_potential.py``. Once you have run ``make``, you may now run the following script.::

    import numpy as np
    import pyvibdmc as dmc
    from pyvibdmc.simulation_utilities import potential_manager as pm

    if __name__ == '__main__': #if using multiprocessing on windows / mac, you need to encapsulate using this line
        pot_dir = 'path/to/Partridge_Schwenke_H2O/' #this directory is part of the one you copied that is outside of pyvibdmc.
        py_file = 'h2o_potential.py'
        pot_func = 'water_pot' # def water_pot(cds) in h2o_potential.py

        #The Potential object assumes you have already made a .so file and can successfully call it from Python
        water_pot = pm.Potential(potential_function=pot_func,
                              python_file=py_file,
                              potential_directory=pot_dir,
                              num_cores=2)
        #optional num_cores parameter for multiprocessing, should not exceed the number of cores on the CPU
        #your machine has. Can use multiprocessing.cpu_count()

        # Starting Structure
        # Equilibrium geometry of water in *atomic units*, then blown up by 1.01 to not start at the bottom of the potential.
        # Can also feed in an entire ensemble of walkers.
        water_coord = np.array([[1.81005599,  0.        ,  0.        ],
                               [-0.45344658,  1.75233806,  0.        ],
                               [ 0.        ,  0.        ,  0.        ]]) * 1.01

        for sim_num in range(5):
            myDMC = dmc.DMC_Sim(sim_name=f"tutorial_water_{sim_num}",
                                  output_folder="tutorial_dmc",
                                  weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                                  num_walkers=8000, #number of geometries exploring the potential surface
                                  num_timesteps=10000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                                  equil_steps=500, #how long before we start collecting wave functions
                                  chkpt_every=500, #checkpoint the simulation every "chkpt_every" time steps
                                  wfn_every=1000, #collect a wave function every "wfn_every" time steps
                                  desc_wt_steps=100, #number of time steps you allow for descendant weighting per wave function
                                  atoms=['H','H','O'],
                                  delta_t=5, #the size of the time step in atomic units
                                  potential=water_pot,
                                  start_structures=np.expand_dims(water_coord,axis=0), #can provide a single geometry, or an ensemble of geometries
                                  masses=None #can put in artificial masses, otherwise it auto-pulls values from the atoms string
            )
            myDMC.run()

Please visit the `API reference <https://pyvibdmc.readthedocs.io/en/latest/autoapi/pyvibdmc/pyvibdmc/index.html#pyvibdmc.pyvibdmc.DMC_Sim>`_
for all the options you may pass the ``DMC_Sim``.

If the simulation dies due to external factors, you may restart a particular DMC simulation using the following code::

    import numpy as np
    import pyvibdmc as dmc
    from pyvibdmc.simulation_utilities import potential_manager as pm

    if __name__ == '__main__': #if using multiprocessing on windows / mac, you need to encapsulate using this line
        # need to reinitalize the water_pot
        pot_dir = 'path/to/Partridge_Schwenke_H2O/' #this directory is the one you copied that is outside of pyvibdmc.
        py_file = 'h2o_potential.py'
        pot_func = 'water_pot' # def water_pot(cds) in h2o_potential.py
        water_pot = pm.Potential(potential_function=pot_func,
                              python_file=py_file,
                              potential_directory=pot_dir,
                              num_cores=2)

        # restart function that reinializes the myDMC object
        # say the 4th [3] simulation died...
        myDMC = dmc.dmc_restart(potential=water_pot,
                                 chkpt_folder='tutorial_dmc',
                                 sim_name='tutorial_water_3')
        myDMC.run()


Once you have run this simulation, you can then analyze the results:

See the `Analyzing DMC Results <https://pyvibdmc.readthedocs.io/en/latest/analysis.html>`_ section.

Tutorial: 1-D Harmonic Oscillator with OH diatomic parameters
---------------------------------------------------------------
``PyVibDMC`` has a Python one-dimensional Harmonic Oscillator potential energy surface built-in as well.
To use it, copy the directory ``pyvibdmc/pyvibdmc/sample_potentials/PythonPots`` outside the directory.
This folder includes ``harmonicOscillator1D.py``. In ``harmonicOscillator1D.py``, there exists the potential
``oh_stretch_harm``, which is the harmonic oscillator potential for an OH stretch.
To use this potential, you have to feed ``'O-H'`` to the ``atoms`` argument, which tells the DMC simulation to use a reduced
mass of an OH diatomic::

    import pyvibdmc as dmc
    from pyvibdmc.simulation_utilities import potential_manager as pm
    import numpy as np

    if __name__ == '__main__': #if using multiprocessing on windows / mac, you need to encapsulate using this line
        pot_dir = 'path/to/PythonPots' #this directory is part of the one you copied that is outside of pyvibdmc.
        py_file = 'harmonicOscillator1D.py'
        pot_func = 'oh_stretch_harm'


        # Equilibrium "geometry" of the 1d harmonic oscillator in *atomic units*,
        # could be blown up (0.8 bohr or something) to not start at the bottom of the potential.
        # harm_coord = np.array([[[0.0]]])
        # or
        # harm_coord = np.zeros((8000,1,1))
        # or
        harm_coord = np.zeros((1,1,1))

        #The Potential object doesn't need a .so file if you are using a python potential
        harm_pot = pm.Potential(potential_function=pot_func,
                                       python_file=py_file,
                                       potential_directory=pot_dir,
                                       num_cores=2
                                )
        #optional num_cores parameter for multiprocessing, should not exceed the number of cores on the CPU
        #your machine has. Can use multiprocessing.cpu_count()
        harm_DMC = dmc.DMC_Sim(sim_name=f"tutorial_HarmOsc_OH_0,
                                  output_folder="tutorial_HarmOsc_dmc",
                                  weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                                  num_walkers=8000, #number of geometries exploring the potential surface
                                  num_timesteps=10000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                                  equil_steps=500, #how long before we start collecting wave functions
                                  chkpt_every=500, #checkpoint the simulation every "chkpt_every" time steps
                                  wfn_every=1000, #collect a wave function every "wfn_every" time steps
                                  desc_wt_steps=100, #number of time steps you allow for descendant weighting per wave function
                                  atoms=['O-H'], #string or list of strings. This will calculate the reduced mass of OH. Can be done with any other diatomic as well.
                                  delta_t=5, #the size of the time step in atomic units
                                  potential=harm_pot,
                                  start_structures=harm_coord,
                                  masses=None #optional parameter for custom masses
                            )
        harm_DMC.run()

One can then, of course, modify the ``harmonicOscillator1D.py`` file in order to include any diatomic they would like using a new python function,
such as N\ :sub:`2` , HCl, or others. You just have to feed in ``'N-N'`` or ``'H-Cl'``, respectively, to ``atoms``. Those examples are
included in the ``.py`` file, feel free to come up with your own!

One can also adjust the frequency in the potential function to be more in line with the frequency of the stretch in the diatomic.