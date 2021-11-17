Potential Energy Surface Interface
=========================================================

PyVibDMC Requires the potential energy surface to be callable from Python.

**Important points and constraints:**

* All calculations inside the DMC simulation are done in atomic units.

* ``PyVibDMC`` itself passes only the coordinates, in Bohr, as a nxmx3 (n=num_geoms,m=num_atoms,3=xyz) NumPy array to the Potential Wrapper ``potential_manager``.

  - To pass more than one argument to the Python function you are using, you will need to create a dictionary that you pass to the potential manager (example below). Otherwise, you will just write a function like: ``def example_potential(cds)``

* ``PyVibDMC`` expects the function you write to return a 1D NumPy array of potential values in Hartree.

Please follow the next steps for calling a PES from Python written in Fortran or C/C++ after a brief note about
parallelization.

Multiproccesing Pool: Parallelizing Potential Calls
-------------------------------------------------------
The ``Potential`` manager uses multiprocessing by default. This allows the user to parallelize the potential call across CPU cores.  This will almost always
yield a very large speed-up, and is highly reccomended. ``PyVibDMC`` uses
Python's `Multiprocessing <https://docs.python.org/3.7/library/multiprocessing.html#module-multiprocessing>`_ module to
do this. The only argument you need to use in order to take advantage of this is the ``num_cores`` parameter::

    from pyvibdmc import potential_manager as pm
    water_pot = pm.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          num_cores=2)

The ``num_cores`` parameter specifies the number of Python processes one would spawn at the beginning of
the DMC simulation. Each Python process takes up 1 core. If you are working with a 16-core CPU,
perhaps use 10 to 12 cores for maximum performance if you are only running one calculation at once.
The number of walkers does NOT need to be divisible by the number of cores/processes.
If this is run on a laptop with 4 cores, only using 2 cores is recommended.

If you explicity do NOT want to use multiprocessing for some reason, such as the parallelization is done elsewhere,
there is a ``Potential_NoMP`` object that you can use instead::

    from pyvibdmc import potential_manager as pm
    water_pot = pm.Potential_NoMP(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          ch_dir=True)

This works identically to ``Potential`` but does not involve the ``multiprocessing`` module at all. Because of how this
code calls the potential, it does not by default call the Python function from the directory of the python function,
instead it calls it from the current working directory that you are running th calculation in. If your python function
relies on the code being in the directory where the python function is housed, you can pass the ``ch_dir=True`` keyword
argument to ``Potential_NoMP``, which will cause the code to chdir into the directory of interest, call the potential,
and then cd back out each time the potential is called. ``ch_dir`` is set to ``False`` by default for computational
efficiency.

Passing more than just the coordinates to the potential manager
------------------------------------------------------------------
The potential wrapper can take in more than just the coordinates if desired.  However, these arguments must be
`pickleable <https://stackoverflow.com/questions/3603581/what-does-it-mean-for-an-object-to-be-picklable-or-pickle-able>`_ due to how multiprocessing works.
The other arguments to be passed to each of the multiprocessing instances should be in the form of a Python dictionary::

    from pyvibdmc.simulation_utilities import potential_manager as pm
    extra_args = {'num_water_molecules': 6, 'other_parameter': True}
    water_pot = pm.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          num_cores=2,
                          pot_kwargs=extra_args)

Then, when writing the Python function::

    def sample_potential(cds, extra_args):
        num_waters = extra_args['num_water_molecules']
        other_param = extra_args['other_parameter']
        ...

In Development: Tensorflow Keras Neural Network Potentials
-------------------------------------------------------
It is possible to run a DMC simulation using a potential energy surface generated using neural networks.  If you have
a tensorflow keras model trained and ready to go, you can plug it in to the potential manager's ``NN_Potential`` object::

    from pyvibdmc import potential_manager as pm
    from pyvibdmc.simulation_utilities.tensorflow_descriptors.tf_coulomb import TF_Coulomb #experimental cds <--> descriptor module
    import tensorflow as tf
    coulomb_transformer = TF_Coulomb([8,1,1,8,1,1]) # water dimer
    extra_args = {'descriptorizer': coulomb_transformer, 'batch_size': 100000}
    my_nn_model = tf.keras.models.load_model('/path/to/model')
    water_pot = pm.NN_Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          model=my_nn_model,
                          pot_kwargs=extra_args)

Then, for the potential call::

    #some other py_file
    def call_nn_potential(cds, model, extra_args):
        batch_size = extra_args['batch_size']
        descriptor = extra_args['descriptorizer']
        transformed_cds = descriptor.get_coulomb(cds)
        y = model.predict(transformed_cds,batch_size=batch_size)
        ...
        return v

This was written with the intention that these models would be evaluated on GPUs, so ``NN_Potential`` does not have multiprocessing
support and is actually a subclass of ``Potential_NoMP``. As such, the ``extra_args`` here do not have to be pickleable.
The potential wrapper is not confined to tensorflow, however other ML packages have not been tested within the confines of ``PyVibDMC``

Fortran Potentials: F2PY
-------------------------------------------------------
`F2PY <https://numpy.org/doc/stable/f2py/>`_ is an easy-to-use interface generator
that calls Fortran from Python. It is baked into NumPy, a prerequisite for this ``PyVibDMC``.
Here is the ``Makefile`` that comes with ``PyVibDMC`` that uses f2py::

   main :
       gfortran -c -fPIC h2opes_v2.f
       gfortran -c -fPIC calc_h2o_pot.f
       python3 -m numpy.f2py -c -I. h2opes_v2.o -m h2o_pot calc_h2o_pot.f

   clean :
       rm *.o *.so

Essentially, we build a Python module in the form of a ``.so`` (shared object/library) file.
The subroutine ``h2o_pot`` in ``calc_h2o_pot.f`` calls a subroutine in ``h2opes_v2.f``, so it is
necessary to compile ``h2opes_v2.f`` before generating the extension. It is optional to also compile
``calc_h2o_pot.f`` before the ``f2py`` call.

A file called ``h2o_pot.cpython...so`` will be generated.  This Python module is now importable inside Python.
This is done in ``h2o_potential.py``::

   # h2o_potential.py
    from h2o_pot import calc_hoh_pot
    import numpy as np


    def water_pot(cds):
        return calc_hoh_pot(cds, len(cds)) # returns a 1D numpy array of potential energy values in Hartree


    if __name__ == '__main__':
        x = np.random.random((100, 3, 3))
        v = water_pot(x)
        print(v)

This program, if called via ``python h2o_potential.py``, will import the ``.so`` Python module called ``h2o_pot``,
exposing the subroutine ``calc_hoh_pot``.

Now, after we made sure this ran to completion, we can call this potential as done in the tutorial::

    Potential(potential_function='water_pot',
              python_file='h2o_potential.py',
              potential_directory='path/to/Partridge_Schwenke_H2O/',
              num_cores=4)

C/C++ Potentials: ctypes
-------------------------------------------------------
For C/C++ Potentials, we require a bit more legwork on the Python side. We will use
`ctypes <https://docs.python.org/3/library/ctypes.html>`_.
Once you compile a shared object
``.so`` file that calls the potential of interest, using the ``ctypes`` module, we can load in that call in Python.
Say we had a shared library called ``lib_expot.so`` that takes in a pointer to an int, a pointer to a coordinate
array of doubles (num_atomsx3 on Python side), and a pointer to the 1D potential array, which in this case is of len(v)=1.

Here is the example of how to load that in and call it::

   # call_cpot.py
   import ctypes
   from numpy.ctypeslib import ndpointer
   import numpy as np

   def call_a_cpot(cds):
      lib = ctypes.cdll.LoadLibrary("./libexpot.so")
      example_fun = lib.calcpot_
      example_fun.restype = None
      example_fun.argtypes = [ctypes.POINTER(ctypes.c_int),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
      nw = ctypes.c_int32(6) #some integer that needs to be passed to the potential
      v = np.zeros(1)
      vpot = np.zeros(len(cds))
      for num,coord in enumerate(cds):
          v = np.zeros(1)
          example_fun(ctypes.byref(nw),v,coord)
          vpot[num] = v[0]

In this example, all the looping is done on the Python side, and so only one geometry is fed to the
``example_fun`` at a time. Indeed, one could loop over the geometres on the C/C++ side and get a speed-up.

Nonetheless, you may then use this Python function in the ``Potential`` object by doing::

   Potential(potential_function='call_a_cpot',
              python_file='call_cpot.py',
              potential_directory=pot_dir,
              num_cores=4)

You don't need to load the shared object file each time, though, thanks to the ``pot_kwargs`` option::

    #before passing to the potential manager...
    lib = ctypes.cdll.LoadLibrary("./libexpot.so")
    example_fun = lib.calcpot_
    example_fun.restype = None
    example_fun.argtypes = [ctypes.POINTER(ctypes.c_int),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    my_kwargs = {'example_fun': example_fun, 'nw': ctypes.c_int32(6)}
    Potential(potential_function='call_a_cpot',
              python_file='call_cpot.py',
              potential_directory=pot_dir,
              num_cores=4,
              pot_kwargs= my_kwargs )

Then...::

    # in the python function call...
    def call_a_cpot(cds, extra_args):
      ex_fun = extra_args['example_fun']
      nw = extra_args['nw']
      v = np.zeros(1)
      vpot = np.zeros(len(cds))
      for num,coord in enumerate(cds):
          v = np.zeros(1)
          ex_fun(ctypes.byref(nw),v,coord)
          vpot[num] = v[0]

MPI4Py: Parallelizing Potential Calls
-------------------------------------------------------

If one has a Python function that calls the potential, one can also parallelize the calls to it using MPI. This is
handled internally using the ``futures`` module in ``MPI4Py``.  While one can run on multiple cores using multiprocessing,
one can run on multiple cores/nodes in high-performance computing environments using the ``MPI_Potential``::

    from pyvibdmc.simulation_utilities.mpi_potential_manager import MPI_Potential
    # Import is like this rather than import pyvibdmc as pv so that people don't have to have MPI4Py installed

    if __name__=='__main__':
        pot_dir = '.../legacy_mbpol/'
        py_file = 'call_mbpol.py'
        pot_func = 'call_any'

        mbp_dimer_pot = MPI_Potential(potential_function=pot_func,
                              python_file=py_file,
                              potential_directory=pot_dir,
                              pot_kwargs={'nw':2})

As you can see, the syntax for ``MPI_Potential`` is similar to the ``Potential`` manager above. No other work needs
to be done to parallelize the code on the user side, this is all handled internally. The ``num_cores``
argument should not be used, as the ``MPI_Potential`` manager simply looks for the ``MPI.COMM_WORLD.get_size()``
attribute to figure out how many MPI processes to use.

PyVibDMC uses the ``mpi4py.futures`` module for MPI parallelization. This module is a relatively new implementation, and does not handle
memory in the most efficient way. While you should get close to linear scaling with the number of nodes used for calling
the potential, there may be memory / efficiency issues as you scale to larger numbers of nodes.

The more difficult part of the MPI potential manager is setting up the desired MPI environment for the high-performance
computing environment one may want to work on. In the McCoy group, we have `containers on dockerhub <https://hub.docker.com/orgs/mccoygroup>`_
for McCoy group students to use. The dockerfiles for these containers are hosted on GitHub on the
`McCoy Group GitHub page <https://github.com/McCoyGroup/mpi-centos-container>`_.

Of course, one could compile or the load the the appropriate MPI module installed on the supercomputer of interest,
and simply load that before using this feature. If using the ``MPI_Potential`` manager, one should run the code
using ``mpirun``

Here is an example ``sbatch`` file for running a containerized MPI DMC simulation using Singularity on Mox::

    #!/bin/bash
    #SBATCH --job-name=mpi_test
    #SBATCH --partition=ilahie
    #SBATCH --account=ilahie
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=28
    #SBATCH --time=01:00:00
    #SBATCH --mem=120G
    #SBATCH --chdir=.

    module load singularity
    . /.../mcenv.sh
    MCENV_IMAGE=/.../mcenv_centos.sif
    MCENV_PACKAGES_PATH=/.../packages

    module load icc_19-ompi_3.1.4

    mcenv=$(mcenv -e)
    mpirun --mca mpi_warn_on_fork 0 -n $SLURM_NTASKS $mcenv --exec python -m mpi4py.futures pv_mpi_test.py

The ``python -m mpi4py.futures pv_mpi_test.py`` syntax should be used regardless of whether or not one is in a container,
amd ``$SLURM_NTASKS`` here is the number of nodes * the number of tasks per node.

Alternative Approach (Not recommended): executables and subprocess calls
-------------------------------------------------------------------------------
If for some reason these approaches do not meet your needs, you can always write the (nxmx3) geometries to a file, call an
executable that loads in the file, and then reload back in the potential values written to a second file, all in
a Python call. This is not recommended as it will be slow, as hard drive reads/writes are slow (especially if you have
a hard drive vs an SSD).  Nonetheless, here is an example of how to do such a thing::

   #pot_call_exec.py
   import subprocess as sub
   def call_exec(cds):
      exportCoords(cds,'coords.txt') #some function that writes the coordinates to file
      sub.run('./pot_executable',cwd='...',shell=True)
      pots = np.loadtxt('pots.txt')
      return pots

Then, we may use this function in the ``Potential`` object::

   Potential(potential_function='call_exec',
              python_file='pot_call_exec.py',
              potential_directory=pot_dir,
              num_cores=1) #cannot parallelize executables easily using multiprocessing. Can read/write to mutliple files...

