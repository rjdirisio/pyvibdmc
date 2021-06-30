Initial Conditions in DMC
=============================================
Typically, a blown up equillibrium geometry is enough for the DMC to begin sampling the ground state.  However, to aid
in equilibration, we can use more advanced methods.

Permutations of Like Atoms
------------------------------
If you want to start with a ground state geometry that has a random distribution of like atoms, you can do this with the
``InitialConditioner`` object.  For example, if you have a methane molecule, you may want to
start with initial conditions so that there is not a bias in picking where "Hydrogen 1" is versus "Hydrogen 2". This
technique will duplicate the geometry to the specified ensemble size, and for each duplication it will swap like atoms
randomly (if you start with ``["C","H1","H2","H3","H4"]``, this code will randomly permute all Hs to get things like
``["C","H1","H2","H4","H3"]``, ``["C","H4","H3","H2","H1"]``, and ``["C","H3","H1","H2","H4"]``).

You can also use this for selective permutations, in the protonated water dimer (H5O2+), you can just permute the
two hydrogen atoms on either side of the water. You will pass in an list of lists::


    import pyvibdmc as pv
    from pyvibdmc import Constants
    from pyvibdmc import potential_manager as pm

    ch4 = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                    [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                    [1.786540362044548, -1.386051328559878, 0.000000000000000],
                    [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                    [-0.8247121421923925, -0.6295306113384560, -1.775332267901544])

    ch4 = Constants.convert(ch4, "angstroms", to_AU=True)  # To Bohr from angstroms

    atms = ["C", "H", "H", "H", "H"]
    initializer = pv.InitialConditioner(coord=ch4,
                                     atoms=atms,
                                     num_walkers=50000,
                                     technique='permute_atoms',
                                     technique_kwargs={'like_atoms': [[0],[1, 2, 3, 4]],
                                                       'ensemble': None})
    new_coords = initializer.run()


    h5o2 = np.array([[ 2.25486805,  0.        ,  0.        ],
                     [-2.25486805,  0.        ,  0.        ],
                     [ 0.        ,  0.        ,  0.12351035],
                     [-3.01319472,  1.48918379, -0.746598  ],
                     [-3.20158756, -0.45448108,  1.49813637],
                     [ 3.01319472, -1.48918379, -0.746598  ],
                     [ 3.20158756,  0.45448108,  1.49813637]])
    atms = ["O", "O", "H", "H", "H", "H", "H"]

    # 3 and 4 are the hydrogen atoms on one side, the 5 and 6 are on the other. They will not permute with each other,
    # only the two pairs by themselves.
    initializer = InitialConditioner(coord=h5o2,
                                     atoms=atms,
                                     num_walkers=50000,
                                     technique='permute_atoms',
                                     technique_kwargs={'like_atoms': [[0],[1],[2],[3,4],[5,6]],
                                                       'ensemble': None})
    new_coords = initializer.run()

The ``ensemble`` option is for if you want to pass in more than just a duplicated minimum energy structure.
If you have an ensemble and you want to permute its like atoms, you can feed in a ``num_walkers, num_atoms, 3`` array.

Sampling from the Harmonic Ground State
-------------------------------------------------------
To sample from the separable, 3N-6 dimensional Gaussian distribution that is the harmonic ground state, you must
run a (Cartesian) normal mode analysis calculation first.  If you have run any electronic structure calculations before
(Psi4, NWChem, Gaussian, etc.), this is equivalent to running a frequency calculation, just using fitted potential
energy surface rather than an ab initio one.::

    import pyvibdmc as pv
    from pyvibdmc import Constants
    from pyvibdmc import potential_manager as pm

    dxx = 1.e-3 # bohr
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

    harm_h2o = pv.HarmonicAnalysis(eq_geom=geom,
                                 atoms=atms,
                                 potential=partridge_schwenke,
                                 dx=dxx)
    freqs, normal_modes = harm_h2o.run(harm_h2o)
    # Turns of scientific notation
    np.set_printoptions(suppress=True)
    print(f"Freqs (cm-1): {freqs}")

The ``HarmonicAnalysis`` object can also take in the arguments ``points_diag=__`` and ``points_off_diag=__``. This
refers to the number of finite difference points used in the generation of the Hessian matrix. These numbers default to
5 and 3 respectively, meaning that the on-diagonal second derivatives are generated using a 5-point finite difference,
and the off-diagonal mixed derivatives use a 3 point finite difference in both dimensions.  Currently, this code only
supports using 3 or 5 point finite difference for either argument.

The 3N frequencies and normal modes that are returned from the harmonic analysis include the 6 near-zero modes from
the translational and rotational degrees of freedom (this code does not support linear molecules).
From there, you will pass these frequencies and normal modes to the ``InitialConditioner``, which will generate the
desired ensemble of walkers that we will feed into the DMC.::

    # Do initial conditions based on freqs and normal modes
    initializer = pv.InitialConditioner(coord=water_geom,
                                     atoms=atms,
                                     num_walkers=50000,
                                     technique='harmonic_sampling',
                                     technique_kwargs={'freqs': freqs,
                                                       'normal_modes': normal_modes,
                                                       'scaling_factor': 1.5},
                                                       'ensemble': None)
    new_coords = initializer.run()

The ``technique_kwargs`` you see above are all necessary to pass in. The ``scaling_factor`` broadens the 3N-6 dimensional
Gaussian distribution by a uniform factor in all dimensions.  In the case above, it is equivalent to saying the
harmonic frequencies are all divided by 1.5, which will give you a broader distribution that the
walkers will sample from. This technique is described in more detail
`in this paper <https://pubs.acs.org/doi/abs/10.1021/acs.jpca.9b06444>`_.

The ``ensemble`` argument is present so that you can pass in a whole ensemble that will be displaced along those normal
modes randomly if desired.  If left as ``None``, then it will simply duplicate the minimum energy geometry you supplied,
and you can ignore the next code block in the tutorial.

If you feed in a ``num_walkers, num_atoms, 3`` array, you can combine this  with the ``permute_atoms`` method above;
start by randomly displacing along the harmonic ground state, then permuting like atoms: ::

    from pyvibdmc import InitialConditioner, HarmonicAnalysis
    from pyvibdmc import Constants
    from pyvibdmc import potential_manager as pm

    ch4 = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                    [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                    [1.786540362044548, -1.386051328559878, 0.000000000000000],
                    [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                    [-0.8247121421923925, -0.6295306113384560, -1.775332267901544])

    ch4 = Constants.convert(ch4, "angstroms", to_AU=True)  # To Bohr from angstroms

    atms = ["C", "H", "H", "H", "H"]
    # Run harmonic analysis
    freqs, normal_modes = HarmonicAnalysis(...)

    # Then, push the freqs, normal modes, and ensemble to the InitialConditioner
    initializer = InitialConditioner(coord=ch4,
                                     atoms=atms,
                                     num_walkers=50000,
                                     technique='harmonic_sampling',
                                     technique_kwargs={'freqs': freqs,
                                                       'normal_modes': normal_modes,
                                                       'scaling_factor': 1.5},
                                                       'ensemble': None)
    harm_coords = initializer.run()

    # Finally, then permute like atoms for each walker that are now spread along the harmonic ground state.
    initializer = InitialConditioner(coord=ch4,
                                     atoms=atms,
                                     num_walkers=50000,
                                     technique='permute_atoms',
                                     technique_kwargs={'like_atoms': [[0],[1, 2, 3, 4]],
                                                       'ensemble': harm_coords})
    new_coords = initializer.run()


Now, the harmonically-sampled-then-permuted ``new_coords`` are passed to the ``DMC_Sim`` object and used during the DMC run::

    import pyvibdmc as pv
    myDMC = pv.DMC_Sim(sim_name=f"conditioner_{sim_num}",
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

Note, you should only run the harmonic calculation THEN permute, not the other way around. This is because this code
produces the eigenvectors of the Hessian that only correspond to the atom ordering of the non-permuted molecular system.
You can, of course, do either individually or use neither technique before a DMC run.