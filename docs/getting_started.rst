Getting Started
===============

This page details how to get started with `PyVibDMC <https://github.com/rjdirisio/pyvibdmc>`_.
This package is still under development, so clone at your own risk!

Theory of Diffusion Monte Carlo (DMC)
-------------------------------------------------------
If you are not someone with experience with Diffusion Monte Carlo, I suggest reading the
`McCoy group's tutorial on and explanation of DMC <https://mccoygroup.github.io/References/References/Monte%20Carlo%20Methods/DMC.html>`_.


This reference also has links to academic publications that detail the method further.

Installation
------------
To install PyVibDMC, first clone it. Then, ``cd`` into to the project directory.

To do a developmental install:

``pip install -e .``

Dependencies (All pre-installed with Anaconda3)
-------------------------------------------------------
- numpy
- matplotlib
- h5py

Usage
--------
Once installed, Then, one can ``import pyvibdmc`` from any directory.
This example will go over how to run 5 DMC simulations on a water monomer using the Fortran Potential Energy Surface built by Partridge and Schwenke (comes with PyVibDMC)::

    import numpy as np
    import pyvibdmc as dmc
    from pyvibdmc import potential_manager as pm

    pot_dir = 'path/to/Partridge_H2O/'
    py_file = 'callPartridgePot.py'
    pot_func = 'potential'

    #Equilibrium of water in *atomic units*, then blown up by 1.01 to not start at the bottom of the potential.
    water_coord = np.array([[1.81005599,  0.        ,  0.        ],
                           [-0.45344658,  1.75233806,  0.        ],
                           [ 0.        ,  0.        ,  0.        ]]) * 1.01

    #The Potential object assumes you have already made a .so file and can successfully call it from Python
    water_pot = pm.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          pool=2)
    #optional pool parameter for multiprocessing, should not exceed the number of cores
    #your machine has. Can use multiprocessing.cpu_count()
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
    from pyvibdmc import potential_manager as pm

    # need to reinitalize the water_pot
    pot_dir = 'path/to/Partridge_H2O/'
    py_file = 'callPartridgePot.py'
    pot_func = 'potential'
    water_pot = pm.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          pool=2)

    #restart function that reinializes the myDMC object
    myDMC = dmc.DMC_Restart(potential=water_pot,
                                 chkpt_folder="tutorial_dmc",
                                 sim_name='tutorial_water_{3}', #the fourth simulation died  (0,1,2,*3*,4)
                                 time_step=2500) #made it to step 2600, so we have a checkpoint at 2500
    myDMC.run()


Once you have run this simulation, you can then analyze the results:

See the `Analyzing DMC Results <https://pyvibdmc.readthedocs.io/en/latest/analysis.html>`_ section.