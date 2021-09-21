import pytest
import pyvibdmc as pv
import os
import numpy as np


def test_imp_samp_derivs():
    water_coord = np.array([[1.81005599, 0., 0.],
                            [-0.45344658, 1.75233806, 0.],
                            [0., 0., 0.]]) * 1.01
    water_coord = np.tile(water_coord, (1000, 1, 1))
    xx = pv.ChainRuleHelper.dr_dx(water_coord, [[0, 2], [1, 2]])
    x = pv.ChainRuleHelper.d2r_dx2(water_coord, [[0, 2], [1, 2]])
    yy = pv.ChainRuleHelper.dcth_dx(water_coord, [[0, 2, 1]])
    y = pv.ChainRuleHelper.d2cth_dx2(water_coord, [[0, 2, 1]])
    a = pv.ChainRuleHelper.dth_dx(water_coord, [[0, 2, 1]])
    b = pv.ChainRuleHelper.d2th_dx2(water_coord, [[0, 2, 1]])
    dpsi_dr = print('doo this')
    dr_dx = np.concatenate([xx, a])
    print('hi')
    assert True


# def test_dpsi_dx():
#     sim_ex_dir = "imp_samp_results"
#     # initialize potential
#     potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/FortPots/Partridge_Schwenke_H2O/')
#     pyFile = 'h2o_potential.py'
#     potFunc = 'water_pot'
#     harm_pot = pv.Potential(potential_function=potFunc,
#                             python_file=pyFile,
#                             potential_directory=potDir,
#                             num_cores=8)
#
#     water_coord = np.array([[1.81005599, 0., 0.],
#                             [-0.45344658, 1.75233806, 0.],
#                             [0., 0., 0.]]) * 1.01
#     water_coord = np.expand_dims(water_coord, 0)
#     start_cds = np.tile(water_coord, (2000, 1, 1))
#     impo = pv.ImpSampManager(trial_function='trial_wavefunction',
#                                   trial_directory=potDir,
#                                   python_file='h2o_trial.py',
#                                   pot_manager=harm_pot,
#                                   deriv_function='dpsi_dx')
#
#     myDMC = pv.DMC_Sim(sim_name="water_impsamp_test_ana",
#                        output_folder=sim_ex_dir,
#                        weighting='discrete',
#                        num_walkers=2000,
#                        num_timesteps=200,
#                        equil_steps=5,
#                        chkpt_every=10,
#                        wfn_every=10,
#                        desc_wt_steps=5,
#                        atoms=["H", "H", "O"],
#                        delta_t=1,
#                        potential=harm_pot,
#                        imp_samp=impo,
#                        log_every=1,
#                        start_structures=start_cds,
#                        )
#     myDMC.run()
