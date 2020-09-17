<<<<<<< HEAD
=======
Getting Started
===============

This page details how to get started with `PyVibDMC <https://github.com/rjdirisio/pyvibdmc>`_.
This package is still under development, so clone at your own risk!

Dependencies (All pre-installed with Anaconda3)
-------------------------------------------------------
- numpy
- matplotlib
- h5py

Installation
------------
**These instructions assume you have the python package manager** ``conda`` **installed.**

To install PyVibDMC, first clone it. Then, ``cd`` into to the project directory.
to do a developmental install:

``pip install -e .``


Usage
--------
Once installed, Then, one can ``import pyvibdmc`` from any directory.
This example will go over how to run the sample water monomer Fortran Potential::

    import numpy as np
    import pyvibdmc as pvdmc
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
    myDMC = pvdmc.DMC_Sim(sim_name=f"tutorial_water",
                     output_folder="tutorial_dmc",
                     weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                     num_walkers=1000, #number of geometries exploring the potential surface
                     num_timesteps=1000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                     equil_steps=500, #how long before we start collecting wave functions
                     chkpt_every=100, #checkpoint the simulation every "chkpt_every" time steps
                     wfn_every=100, #collect a wave function every "wfn_every" time steps
                     desc_steps=50, #number of time steps you allow for descendant weighting per wave function
                     atoms=['H','H','O'],
                     delta_t=5, #the size of the time step in atomic units
                     potential=water_pot,
                     start_structures=np.expand_dims(water_coord,axis=0), #can provide a single geometry, or an ensemble of geometries
                     masses=None #can put in artificial masses, otherwise it auto-pulls values from the atoms string list
                     )
    myDMC.run()


>>>>>>> analysis
