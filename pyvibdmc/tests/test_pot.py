import os
import time

import pytest
import numpy as np
import pyvibdmc as pv


def test_regular_pot():
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)
    cds = np.random.random((100, 1, 1))
    start = time.time()
    for _ in range(10):
        v = harm_pot.getpot(cds)
    print(time.time() - start)
    assert True


def test_arg_pot():
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm_with_arg'
    pot_dict = {'freq': pv.Constants.convert(4000, 'wavenumbers', to_AU=True),
                'mass': pv.Constants.reduced_mass("O-H", to_AU=True)}
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2,
                            pot_kwargs=pot_dict
                            )
    cds = np.random.random((100, 1, 1))
    start = time.time()
    for _ in range(10):
        v = harm_pot.getpot(cds, pot_dict)
    print(time.time() - start)
    assert True


def test_no_mp_pot():
    """Tests if one can load in a data file when you chdir into the pot directory and call the potential"""
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm_loadtxt'
    harm_pot = pv.Potential_NoMP(potential_function=potFunc,
                                 python_file=pyFile,
                                 potential_directory=potDir,
                                 ch_dir=True)
    cds = np.random.random((100, 1, 1))
    start = time.time()
    for _ in range(10):
        v = harm_pot.getpot(cds)
    print(time.time() - start)
    assert True


def test_nn_pot():
    from ..simulation_utilities.tensorflow_descriptors.distance_descriptors import DistIt
    import tensorflow as tf

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__),
                          '../sample_potentials/TensorflowPots/')  # only necesary for testing
    pyFile = 'call_sample_model.py'
    potFunc = 'sample_h4o2_pot'
    coulomb = DistIt([8, 1, 1] * 2,
                     'coulomb')
    pot_dict = {'descriptor': coulomb,
                'batch_size': 100}
    model_path = f'{potDir}/sample_h4o2_nn.h5'
    model = tf.keras.models.load_model(model_path)
    harm_pot = pv.NN_Potential(potential_function=potFunc,
                               python_file=pyFile,
                               potential_directory=potDir,
                               model=model,
                               pot_kwargs=pot_dict
                               )
    cds = np.random.random((100, 6, 3))
    start = time.time()
    for _ in range(10):
        v = harm_pot.getpot(cds)
    print(time.time() - start)
    assert True


def test_pot_direct():
    def dummy_pot(coords):
        return np.repeat(500,len(coords))

    pot = pv.Potential_Direct(potential_function=dummy_pot)
    cds = np.random.random((100,10,3))
    vs = pot.getpot(cds)
    assert len(vs) == len(cds)


def test_mpi_pot():
    from pyvibdmc.simulation_utilities.mpi_potential_manager import MPI_Potential
    cdz = np.random.random((100, 1, 1))
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm_with_arg'
    ex_arg = {'mass': 1.0, 'freq': 1.0}
    mpi = MPI_Potential(potential_function=potFunc,
                           potential_directory=potDir,
                           python_file=pyFile,
                           pot_kwargs=ex_arg)
    mpi.getpot(cdz)
