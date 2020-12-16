Initial Conditions in DMC
=============================================
Typically, a blown up equillibrium geometry is enough for the DMC to begin sampling the ground state.  However, to aid
in equilibration, we can use more advanced methods.

Sampling from the Harmonic Ground State
-------------------------------------------------------
To sample from the separable, 3N-6 dimensional Gaussian distribution that is the harmonic ground state, you must
run a (Cartesian) normal mode analysis calculation first.  If you have run any electronic structure calculations before
(Psi4, NWChem, Gaussian, etc.), this is equivalent to running a frequency calculation, just on a fitted potential
energy surface rather than an ab initio one.

This is done within ``PyVibDMC`` using the
``pyvibdmc.simulation_utilities.initial_conditioner``::

    from pyvibdmc.simulation_utilities.initial_conditioner import *
    from pyvibdmc.simulation_utilities import Constants
    from pyvibdmc.simulation_utilities import potential_manager as pm

    dxx = 1.e-3
    water_geom = np.array([[0.9578400, 0.0000000, 0.0000000],
                           [-0.2399535, 0.9272970, 0.0000000],
                           [0.0000000, 0.0000000, 0.0000000]])
    # Everything is in  Atomic Units going into generating the Hessian.
    pot_dir = path/to/Partridge_Schwenke_H2O/')
    py_file = 'h2o_potential.py'
    pot_func = 'water_pot'
    partridge_schwenke = pm.Potential(potential_function=pot_func,
                                      potential_directory=pot_dir,
                                      python_file=py_file,
                                      num_cores=1)
    geom = Constants.convert(water_geom, "angstroms", to_AU=True)  # To Bohr from angstroms
    atms = ["H", "H", "O"]

    harm_h2o = harmonic_analysis(eq_geom=geom,
                                 atoms=atms,
                                 potential=partridge_schwenke,
                                 dx=dxx)
    freqs, normal_modes = harmonic_analysis.run(harm_h2o)
    # Turns of scientific notation
    np.set_printoptions(suppress=True)
    print(f"Freqs (cm-1): {freqs}")

The ``harmonic_analysis`` object can also take in the arguments ``points_diag=__`` and ``points_off_diag=__``. This
refers to the number of finite difference points used in the generation of the Hessian matrix. These numbers default to
5 and 3 specifically, meaning that the on-diagonal second derivatives are generated using a 5-point finite difference,
and the mixed derivatives use a 3 point finite difference in both dimensions.  Currently, this code only supports using
3 or 5 point finite difference for either argument.

The 3N frequencies and normal modes that are returned from the harmonic analysis include the 6 near-zero modes from
the translational and rotational degrees of freedom (this code does not support linear molecules).
From there, you will pass these frequencies and normal modes to the ``InitialConditioner``, which will generate the
desired ensemble of walkers that we will feed into the DMC.::

    # Do initial conditions based on freqs and normal modes
    initializer = InitialConditioner(coord=water_geom,
                                     atoms=atms,
                                     num_walkers=50000,
                                     technique='harmonic_sampling',
                                     technique_kwargs={'freqs': freqs,
                                                       'normal_modes': normal_modes,
                                                       'scaling_factor': 1.5})
    new_coords = initializer.run()

The ``technique_kwargs`` you see above are all necessary to pass in. The ``scaling_factor`` broadens the 3N-6 dimensional
Gaussian distribution by a uniform factor in all dimensions.  In the case above, it is equivalent to saying the
harmonic frequencies are all divided by 1.5, which will give you a broader distribution that the
walkers will sample from. This technique is described in more detail
`in this paper <https://pubs.acs.org/doi/abs/10.1021/acs.jpca.9b06444>`_.

Now, the new_coords are passed to the ``DMC_Sim`` object and used during the DMC run::

    myDMC = dmc.DMC_Sim(sim_name=f"conditioner_{sim_num}",
                                  output_folder="initial_conditions_tutorial",
                                  weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                                  num_walkers=50000, #number of geometries exploring the potential surface
                                  num_timesteps=10000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                                  equil_steps=500, #how long before we start collecting wave functions
                                  chkpt_every=500, #checkpoint the simulation every "chkpt_every" time steps
                                  wfn_every=1000, #collect a wave function every "wfn_every" time steps
                                  desc_wt_steps=100, #number of time steps you allow for descendant weighting per wave function
                                  atoms=['H','H','O'],
                                  delta_t=10, #the size of the time step in atomic units
                                  potential=water_pot,
                                  start_structures=new_coords,
                                  masses=None #can put in artificial masses, otherwise it auto-pulls values from the atoms string
            )

Permutations of Like Atoms
------------------------------
Documentation and implementation pending.