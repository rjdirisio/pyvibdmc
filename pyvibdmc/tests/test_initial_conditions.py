import numpy as np
import pytest
import sys, os

from ..analysis import *
from ..simulation_utilities.initial_conditioner import *

"""
These tests can be run locally, if needed, after compilining the Partridge_Scwhwenke potential manually. 
I did not want to go through the trouble of installing gfortran during Travis CI, which is different for 
mac and linux environments.
"""


def test_pyvibdmc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyvibdmc" in sys.modules


# def test_harm_analysis():
#     dxx = 1.e-3
#     water_geom = np.array([[0.9578400, 0.0000000, 0.0000000],
#                            [-0.2399535, 0.9272970, 0.0000000],
#                            [0.0000000, 0.0000000, 0.0000000]])
#     # Everything is in  Atomic Units going into generating the Hessian.
#     pot_dir = os.path.join(os.path.dirname(__file__), '../sample_potentials/FortPots/Partridge_Schwenke_H2O/')
#     py_file = 'h2o_potential.py'
#     pot_func = 'water_pot'
#     partridge_schwenke = pm.Potential(potential_function=pot_func,
#                                       potential_directory=pot_dir,
#                                       python_file=py_file,
#                                       num_cores=1)
#     geom = Constants.convert(water_geom, "angstroms", to_AU=True)  # To Bohr from angstroms
#     atms = ["H", "H", "O"]
#
#     harm_h2o = harmonic_analysis(eq_geom=geom,
#                                  atoms=atms,
#                                  potential=partridge_schwenke,
#                                  dx=dxx)
#     freqs, normal_modes = harmonic_analysis.run(harm_h2o)
#     # Turns of scientific notation
#     np.set_printoptions(suppress=True)
#     print(f"Freqs (cm-1): {freqs}")
#
#
# def test_initial_conditions():
#     pot_dir = os.path.join(os.path.dirname(__file__), '../sample_potentials/FortPots/Partridge_Schwenke_H2O/')
#     py_file = 'h2o_potential.py'
#     pot_func = 'water_pot'
#     partridge_schwenke = pm.Potential(potential_function=pot_func,
#                                       potential_directory=pot_dir,
#                                       python_file=py_file,
#                                       num_cores=1)
#
#     water_geom = np.array([[0.9578400, 0.0000000, 0.0000000],
#                            [-0.2399535, 0.9272970, 0.0000000],
#                            [0.0000000, 0.0000000, 0.0000000]])
#
#     water_geom = Constants.convert(water_geom, "angstroms", to_AU=True)  # To Bohr from angstroms
#
#     atms = ["H", "H", "O"]
#
#     # Do harmonic analysis
#     print("Running harmonic analysis...")
#     ha = harmonic_analysis(eq_geom=water_geom,
#                            atoms=atms,
#                            potential=partridge_schwenke,
#                            )
#
#     freqz, nmz = ha.run()
#     print("Done with harmonic analysis...")
#     print(f"Harmonic Frequencies: {freqz}")
#
#     # Do initial conditions based on freqs and normal modes
#     initializer = InitialConditioner(coord=water_geom,
#                                      atoms=atms,
#                                      num_walkers=5000,
#                                      technique='harmonic_sampling',
#                                      technique_kwargs={'freqs': freqz,
#                                                        'normal_modes': nmz,
#                                                        'scaling_factor': 1})
#     new_coords = initializer.run()
#
#     # Check that new coords actually sampled harmonic g.s.
#     Constants.convert(new_coords, 'angstroms', to_AU=False)
#     harmz = AnalyzeWfn(new_coords)
#     oh1 = harmz.bond_length(0, 2)
#     oh2 = harmz.bond_length(1, 2)
#     bend = harmz.bond_angle(0, 2, 1)
#     bend = np.degrees(bend)
#     xy = AnalyzeWfn.projection_1d((1 / np.sqrt(2)) * (oh1 + oh2),
#                                   desc_weights=np.ones(len(oh1)),
#                                   range=(1, 4),
#                                   bin_num=30)
#     Plotter.plt_hist1d(xy, xlabel='Symm Stretch', save_name='symm_str.png')
#     xy = AnalyzeWfn.projection_1d((1 / np.sqrt(2)) * (oh1 - oh2),
#                                   desc_weights=np.ones(len(oh1)),
#                                   range=(-1.5, 1.5),
#                                   bin_num=30)
#     Plotter.plt_hist1d(xy, xlabel='Anti Stretch', save_name='anti_str.png')
#
#     xy = AnalyzeWfn.projection_1d(bend,
#                                   desc_weights=np.ones(len(bend)),
#                                   range=(40, 170),
#                                   bin_num=30)
#     Plotter.plt_hist1d(xy, xlabel='Bend (Degrees)', save_name='bend.png')
#     assert True