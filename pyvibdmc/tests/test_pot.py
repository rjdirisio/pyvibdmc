"""
Unit and regression test for the pyvibdmc package.

"""
import numpy as np
# Import package, test suite, and other packages as needed

from ..simulation_utilities import *
from ..analysis import *
import pytest
import os
import time

def test_regular_pot():
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    harm_pot = Potential(potential_function=potFunc,
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
    pot_dict = {'freq': Constants.convert(4000, 'wavenumbers', to_AU=True),
                'mass': Constants.reduced_mass("O-H",to_AU=True)}
    harm_pot = Potential(potential_function=potFunc,
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

def test_nn_pot():
    from ..simulation_utilities.tensorflow_descriptors.tf_coulomb import TF_Coulomb
    import tensorflow as tf

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/TensorflowPots/')  # only necesary for testing
    pyFile = 'call_sample_model.py'
    potFunc = 'sample_h4o2_pot'
    coulomb = TF_Coulomb([8, 1, 1, 8, 1, 1])
    pot_dict = {'descriptor': coulomb,
                'batch_size': 100}
    model_path = f'{potDir}/sample_h4o2_nn.h5'
    model = tf.keras.models.load_model(model_path)
    harm_pot = NN_Potential(potential_function=potFunc,
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