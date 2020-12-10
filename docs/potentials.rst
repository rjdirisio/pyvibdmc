Potential Energy Surface Interface
=========================================================

PyVibDMC Requires the potential energy surface to be callable from Python.

**Some important points and constraints:**

- All calculations inside the DMC simulation are done in atomic units.

- ``PyVibDMC`` passes only the coordinates, in Bohr, as a nxmx3 (n=num_geoms,m=num_atoms,3=xyz) NumPy array to the Python potential function. Your function must only take in coords as an argument ``def example_call(coords):`` All other attributes you must pass to the Fortran/C side must be done inside your python function.

- ``PyVibDMC`` expects the function to return a 1D NumPy array of potential values in Hartree.

Please follow the next steps for calling a PES from Python written in Fortran or C/C++ after a brief note about
parallelization.

Multiproccesing Pool: Parallelizing Potential Calls
-------------------------------------------------------
The ``Potential`` manager has an option to parallelize the potential call across CPU cores.  This will almost always
yield a very large speed-up, and is highly reccomended. ``PyVibDMC`` uses
Python's `Multiprocessing <https://docs.python.org/3.7/library/multiprocessing.html#module-multiprocessing>`_ module to
do this. The only argument you need to use in order to take advantage of this is the ``num_cores`` parameter::

    water_pot = pm.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          num_cores=2)

The ``num_cores`` parameter specifies the number of Python processes one would spawn at the beginning of
the DMC simulation. Each Python process takes up 1 core. If you are working with a 16-core CPU,
perhaps use 10 to 12 cores for maximum performance if you are only running one calculation at once.
The number of walkers does NOT need to be divisible by the number of cores/processes.
If this is run on a laptop with 4 cores, only using 2 cores is recommended.

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
        return calc_hoh_pot(cds, len(cds))


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
      lib = ctypes.cdll.LoadLibrary("./lib_expot.so")
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

Alternative Approach (Not recommended): executables and subprocess calls
-------------------------------------------------------------------------------
If for some reason these do not meet your needs, you can always write the (nxmx3) geometries to a file, call an
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

