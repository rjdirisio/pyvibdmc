Guided DMC / Importance Sampling
========================================

PyVibDMC supports the use of guiding functions to reduced the required number of walkers to obtain a reasonable result
from DMC simulations. These guiding functions must be supplied by the user.

Theory of Importance Sampling
-------------------------------------------------------
The theory of importance sampling in the context of vibrational Diffusion Monte Carlo is outlined in
the supplementary information of a paper by `Lee, Madison, and McCoy <https://pubs.acs.org/doi/abs/10.1021/acs.jpca.8b11213>`_.

Further applications can be found in papers by `Finney, DiRisio, and McCoy <https://pubs.acs.org/doi/10.1021/acs.jpca.0c07181>`_ as well as
`Lee, Vetterli, Boyer, and McCoy <https://pubs.acs.org/doi/full/10.1021/acs.jpca.0c05686?ref=recommended>`_.

Using Guiding functions in PyVibDMC
----------------------------------------------
The guiding function interface is quite similar to the potential energy interface for PyVibDMC. The user *must* provide
a Python function that takes in the ``num_walkers x num_atoms x 3`` walker array and evaluate the value of the trial
wave function for each walker. The value of the trial wave function should be returned as a
NumPy array of size ``num_walkers``.  The user may also pass along ``trial_kwargs`` in the form of a dictionary, much
like what can be done for `the potential energy interface <https://pyvibdmc.readthedocs.io/en/latest/potentials.html#passing-more-than-just-the-coordinates-to-the-potential-manager>`_.

Optionally, the user may also provide two other Python functions that calculate the first and second
derivatives of the wave function with respect to the ``3N`` Cartesian coordinates. The functions should return
``num_walkers x num_atoms x 3`` NumPy arrays that correspond to the value of the derivatives of each walker
in the x,y, and z components of the various atoms in the molecular system. If no first and second derivatives are
provided, PyVibDMC defaults to computing the derivatives numerically using finite difference. If a first derivative is
provided but no second derivative (or vice versa), the first derivative is calculated using the function and the second derivative
is calculated using finite difference.

IMPORTANT REQUIREMENTS:

* The potential manager is responsible for setting up the parallelization in the importance sampling. If one is using multiprocessing in the potential manager, the importance sampling will be parallelized using multiprocessing, MPI for MPI, and single-core for single-core.

  * The one exception to this rule is when using a NN_Potential. One can pass the ``new_pool_num_cores`` argument in order to use ImpSampManager if using Potential_NoMP or NN_Potential.

* The Python file that holds the function that calculates the trial wave function (and optionally the derivatives) MUST be in the same directory as the Python file that calls the potential energy surface. This is a restrictive measure that was put in to make calls cleaner inside PyVibDMC, as sometimes the current working directory matters when one loads in data files on-the-fly and things of the like.

There are three importance sampling managers: ``ImpSampManager``, ``ImpSampManager_NoMP``, and ``MPI_ImpSampManager``. The first two can
easily be used with PyVibDMC::

    import pyvibdmv as pv
    pot_func = ...
    py_file = ...
    pot_dir = ...

    """
    # No multiprocessing at all.
    water_pot = pv.Potential_NoMP(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          chdir=False)
    water_imp = pv.ImpSampManager_NoMP(trial_function,
                         trial_directory, #same as potential_directory if using MP or MPI
                         python_file,
                         pot_manager=water_pot,
                         chdir=..., #optional
                         deriv_function=..., #optional
                         s_deriv_function=..., #optional
                         trial_kwargs=..., #May pass a dict with important things to trial function call
                         deriv_kwargs=..., #May pass a dict with important things to trial function call - only use if deriv_function is set to something
                         s_deriv_kwargs=...) # see deriv_kwargs
    """

    # Using multiprocessing for potential and imp samp
    water_pot = pv.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          num_cores=2) #potential and impsamp will both use the same pool of workers

    water_imp = pv.ImpSampManager(trial_function,
                         trial_directory, #same as potential_directory
                         python_file,
                         pot_manager=water_pot,
                         deriv_function=..., #optional string like trial_function
                         s_deriv_function=..., #optional string like trial_function
                         trial_kwargs=..., #May pass a dict with important things to trial function call
                         deriv_kwargs=..., #May pass a dict with important things to deriv function call - only use if deriv_function is set to something
                         s_deriv_kwargs=...) # see deriv_kwargs

    # pass to DMC_Sim
    my_sim = pv.DMC_sim(...,
                        imp_samp=water_imp,
                        # imp_samp_oned=False, # only true if DMC sim is on 1D problem.
                                               # Defaults to False.
                        ...)

Examples of a trial wavefunction (and derivatives) can be found in the Partridge Schwenke sample potential ``h2o_trial.py`` or the
harmonic oscillator sample potential ``harm_trial_wfn.py``.

The MPI version of the ImpSampManager can be used in the same way as above, except one must use an
``MPI_Potential`` object for the potential manager and and ``MPI_ImpSampManager`` object::

    import pyvibdmc as pv
    from pyvibdmc.simulation_utilities.mpi_potential_manager import MPI_Potential
    from pyvibdmc.simulation_utilities.mpi_imp_samp_manager import MPI_ImpSampManager
    ...
    # Must import this way in order to access the MPI modules.
    mpi_pot = MPI_Potential(...)
    mpi_imp = MPI_ImpSampManager(...)
    my+sim = pv.DMC_Sim(...,
                        imp_samp=mpi_imp,
                        ...)


Chain rule helper
----------------------------------------------

Coming soon...