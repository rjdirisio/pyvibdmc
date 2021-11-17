import sys, os

import pytest
import numpy as np
import pyvibdmc as pv

"""
These tests can be run locally, if needed, after compilining the Partridge_Scwhwenke potential manually. 
I did not want to go through the trouble of installing gfortran during Travis CI, which is different for 
mac and linux environments.
"""


def test_initial_conditions_premute():
    ch5 = np.array([[1.000000000000000, 2.000000000000000, 3.000000000000000],
                    [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                    [1.786540362044548, -1.386051328559878, 0.000000000000000],
                    [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                    [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                    [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
    atms = ["C", "H", "H", "H", "H", "H"]
    ch5 = np.expand_dims(ch5, 0)
    initializer = pv.InitialConditioner(coord=ch5,
                                        atoms=atms,
                                        num_walkers=5000,
                                        technique='permute_atoms',
                                        technique_kwargs={'like_atoms': [[0], [1, 2, 3, 4, 5]],
                                                          'ensemble': None})
    permuted_coords = initializer.run()
    assert True


# Uncomment once you compile the PS surface
# def test_harm_analysis():
#     dxx = 1.e-3
#     water_geom = np.array([[0.9578400, 0.0000000, 0.0000000],
#                            [-0.2399535, 0.9272970, 0.0000000],
#                            [0.0000000, 0.0000000, 0.0000000]])
#     # Everything is in  Atomic Units going into generating the Hessian.
#     pot_dir = os.path.join(os.path.dirname(__file__), '../sample_potentials/FortPots/Partridge_Schwenke_H2O/')
#     py_file = 'h2o_potential.py'
#     pot_func = 'water_pot'
#     partridge_schwenke = pv.Potential(potential_function=pot_func,
#                                       potential_directory=pot_dir,
#                                       python_file=py_file,
#                                       num_cores=1)
#     geom = pv.Constants.convert(water_geom, "angstroms", to_AU=True)  # To Bohr from angstroms
#     atms = ["H", "H", "O"]
#
#     harm_h2o = pv.HarmonicAnalysis(eq_geom=geom,
#                                    atoms=atms,
#                                    potential=partridge_schwenke,
#                                    dx=dxx)
#     freqs, normal_modes = pv.HarmonicAnalysis.run(harm_h2o)
#     # Turns of scientific notation
#     np.set_printoptions(suppress=True)
#     print(f"Freqs (cm-1): {freqs}")
#
# def test_harm_dip():
#     dxx = 1.e-3
#     tetramer = np.array([[-1.55099190, 1.94067311, 0.14704161],
#                          [-0.91203521, -2.30896272, 0.14764850],
#                          [2.46079102, 0.36718848, 0.14815394],
#                          [0.00253217, 0.00164013, -0.47227522],
#                          [-1.96589559, 2.46292466, -0.54312627],
#                          [-2.13630186, 1.99106023, 0.90604777],
#                          [-1.18003749, -2.92157144, -0.54090532],
#                          [-0.65410291, -2.84939169, 0.89772271],
#                          [2.79828182, 0.87002791, 0.89281564],
#                          [3.12620054, 0.43432898, -0.54032031],
#                          [-0.31106354, -0.91215572, -0.20184621],
#                          [0.95094197, 0.18695800, -0.20259538],
#                          [-0.63272209, 0.72926470, -0.20069859]])
#     tetramer = pv.Constants.convert(tetramer, 'angstroms', to_AU=True)
#
#     # Everything is in  Atomic Units going into generating the Hessian.
#     pot_dir = '/home/netid.washington.edu/rjdiri/Documents/Potentials/Bowman_prot_pot/'
#     py_file = 'call_tet.py'
#     pot_func = 'call_pot'
#     pot = pv.Potential_NoMP(potential_function=pot_func,
#                             potential_directory=pot_dir,
#                             python_file=py_file,
#                             ch_dir=True)
#
#     pot_dir = '/home/netid.washington.edu/rjdiri/Documents/Potentials/bigProtClusterPotentials_pacthed/big1'
#     py_file = 'call_tet_dip.py'
#     pot_func = 'call_dipz'
#     dip = pv.Potential_NoMP(potential_function=pot_func,
#                             potential_directory=pot_dir,
#                             python_file = py_file,
#                             ch_dir=True)
#
#     atms = ["O","O","O","O","H","H","H","H","H","H","H","H","H",]
#
#     harm_h2o = pv.HarmonicAnalysis(eq_geom=tetramer,
#                                    atoms=atms,
#                                    potential=pot,
#                                    dx=dxx,
#                                    dipole=dip)
#     freqs, normal_modes = harm_h2o.run()
#     dipz = harm_h2o.dipole_derivs()
#     # Turns of scientific notation
#     np.set_printoptions(suppress=True)
#     print(f"Freqs (cm-1): {freqs}")
#
#
#
# def test_initial_conditions():
#     pot_dir = os.path.join(os.path.dirname(__file__), '../sample_potentials/FortPots/Partridge_Schwenke_H2O/')
#     py_file = 'h2o_potential.py'
#     pot_func = 'water_pot'
#     partridge_schwenke = pv.Potential(potential_function=pot_func,
#                                       potential_directory=pot_dir,
#                                       python_file=py_file,
#                                       num_cores=1)
#
#     water_geom = np.array([[0.9578400, 0.0000000, 0.0000000],
#                            [-0.2399535, 0.9272970, 0.0000000],
#                            [0.0000000, 0.0000000, 0.0000000]])
#
#     water_geom = pv.Constants.convert(water_geom, "angstroms", to_AU=True)  # To Bohr from angstroms
#
#     atms = ["H", "H", "O"]
#
#     # Do harmonic analysis
#     print("Running harmonic analysis...")
#     ha = pv.HarmonicAnalysis(eq_geom=water_geom,
#                              atoms=atms,
#                              potential=partridge_schwenke,
#                              )
#
#     freqz, nmz = ha.run()
#     print("Done with harmonic analysis...")
#     print(f"Harmonic Frequencies: {freqz}")
#
#     # Do initial conditions based on freqs and normal modes
#     initializer = pv.InitialConditioner(coord=water_geom,
#                                         atoms=atms,
#                                         num_walkers=5000,
#                                         technique='harmonic_sampling',
#                                         technique_kwargs={'ensemble': None,
#                                                           'freqs': freqz,
#                                                           'normal_modes': nmz,
#                                                           'scaling_factor': 1})
#     new_coords = initializer.run()
#
#     # Check that new coords actually sampled harmonic g.s.
#     pv.Constants.convert(new_coords, 'angstroms', to_AU=False)
#     harmz = pv.AnalyzeWfn(new_coords)
#     oh1 = harmz.bond_length(0, 2)
#     oh2 = harmz.bond_length(1, 2)
#     bend = harmz.bond_angle(0, 2, 1)
#     bend = np.degrees(bend)
#     xy = pv.AnalyzeWfn.projection_1d((1 / np.sqrt(2)) * (oh1 + oh2),
#                                      desc_weights=np.ones(len(oh1)),
#                                      range=(1, 4),
#                                      bins=30)
#     pv.Plotter.plt_hist1d(xy, xlabel='Symm Stretch', save_name='symm_str.png')
#     xy = pv.AnalyzeWfn.projection_1d((1 / np.sqrt(2)) * (oh1 - oh2),
#                                      desc_weights=np.ones(len(oh1)),
#                                      range=(-1.5, 1.5),
#                                      bins=30)
#     pv.Plotter.plt_hist1d(xy, xlabel='Anti Stretch', save_name='anti_str.png')
#
#     xy = pv.AnalyzeWfn.projection_1d(bend,
#                                      desc_weights=np.ones(len(bend)),
#                                      range=(40, 170),
#                                      bins=30)
#     pv.Plotter.plt_hist1d(xy, xlabel='Bend (Degrees)', save_name='bend.png')
#     assert True
