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
    yy = pv.ChainRuleHelper.dcth_dx(water_coord, [[0, 1, 2]])
    y = pv.ChainRuleHelper.d2cth_dx2(water_coord, [[0, 1, 2]])
    a = pv.ChainRuleHelper.dth_dx(water_coord, [[0, 1, 2]])
    b = pv.ChainRuleHelper.d2th_dx2(water_coord, [[0, 1, 2]])
    dpsi_dr = print('doo this')
    dr_dx = np.concatenate([xx, a])
    print('hi')


def test_dpsi_dx():
    sim_ex_dir = "imp_samp_results"
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/FortPots/Partridge_Schwenke_H2O/')
    # purposes
    pyFile = 'h2o_potential.py'
    potFunc = 'water_pot'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    water_coord = np.array([[1.81005599, 0., 0.],
                            [-0.45344658, 1.75233806, 0.],
                            [0., 0., 0.]]) * 1.01
    water_coord -= 1
    th1 = pv.MolRotator.gen_rot_mats(np.repeat(np.deg2rad(75),2001),0)
    th2 = pv.MolRotator.gen_rot_mats(np.repeat(np.deg2rad(30),2001),1)
    start_cds = np.tile(water_coord, (2001, 1, 1))  # Make it (1 x num_atoms x 3)
    stretch = np.linspace(-0.5, 0.5, num=2001)
    start_cds[:, 0, 0] += stretch
    start_cds = pv.MolRotator.rotate_geoms(th1,start_cds)
    start_cds = pv.MolRotator.rotate_geoms(th2,start_cds)
    # np.save("start_cds.npy",start_cds)
    #
    # water_coord = np.array([[1.81005599, 0., 0.],
    #                         [-0.45344658, 1.75233806, 0.],
    #                         [0., 0., 0.]]) * 1.01
    # water_coord = np.expand_dims(water_coord,0)
    # start_cds = np.tile(water_coord,(2001,1,1))
    impo = pv.ImpSampManager_NoMP(trial_function='trial_wavefunction',
                                  trial_directory=potDir,
                                  python_file='h2o_trial.py',
                                  deriv_function='dpsi_dx')

    myDMC = pv.DMC_Sim(sim_name="water_impsamp_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=2001,
                       num_timesteps=1000,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["H", "H", "O"],
                       delta_t=1,
                       potential=harm_pot,
                       imp_samp=impo,
                       log_every=1,
                       start_structures=start_cds,
                       )
    myDMC.run()
