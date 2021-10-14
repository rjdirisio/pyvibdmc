import pytest
import pyvibdmc as pv
import os
import numpy as np


def test_imp_samp_derivs():
    water_coord = np.array([[1.81005599, 0., 0.],
                            [-0.45344658, 1.75233806, 0.],
                            [0., 0., 0.]]) * 1.01
    water_coord = np.tile(water_coord, (1000, 1, 1))

    ohs = [[0,2],[1,2]]
    hoh = [0,2,1]
    crh = pv.ChainRuleHelper(water_coord,np)
    dr_dxs = [crh.dr_dx(oh) for oh in ohs]
    d2r_dx2s = [crh.d2r_dx2(oh) for oh in ohs]
    d2r_dx2s = [crh.d2r_dx2(oh,dr_dx=dr_dxs[num]) for num, oh in enumerate(ohs)]
    dcth_dx = crh.dcth_dx(hoh)
    dcth_dx = crh.dcth_dx(hoh,
                          dr_da=dr_dxs[0],
                          dr_dc=dr_dxs[1])
    
    dth_dx = crh.dth_dx(hoh)
    dth_dx = crh.dth_dx(hoh,
                        dcth_dx=dcth_dx,
                        dr_da=dr_dxs[0],
                        dr_dc=dr_dxs[1])

    d2th_dx2 = crh.d2th_dx2(hoh)
    d2th_dx2 = crh.d2th_dx2(hoh,
                            dcth_dx=dcth_dx,
                            dr_da=dr_dxs[0],
                            dr_dc=dr_dxs[1],
                            d2r_da2=d2r_dx2s[0],
                            d2r_dc2=d2r_dx2s[0])
    print('hi')
    assert True

#
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
#
# def test_hex():
#     sim_ex_dir = "imp_samp_results"
#     # initialize potential
#     potDir = '/home/netid.washington.edu/rjdiri/Documents/Potentials/legacy_mbpol'
#     pyFile = 'call_mbpol.py'
#     potFunc = 'call_hexamer'
#     harm_pot = pv.Potential(potential_function=potFunc,
#                             python_file=pyFile,
#                             potential_directory=potDir,
#                             num_cores=8)
#
#     hex = np.array([[0.80559297, 1.82637417, 0.19044583],
#                     [1.64546268, 1.33062728, 0.20230004],
#                     [1.03131975, 2.74531261, 0.3303837],
#                     [-0.86971419, -0.05280485, 1.64663647],
#                     [-0.40947453, 0.75209702, 1.37618396],
#                     [-1.70683682, -0.02424652, 1.15831962],
#                     [0.65167739, -1.73597316, 0.2335045],
#                     [0.05821864, -1.2362209, 0.84210027],
#                     [0.569203, -2.6591634, 0.4706903],
#                     [-0.51396268, 0.08861126, -1.76674358],
#                     [-0.09074241, 0.82334616, -1.30525568],
#                     [-0.09916254, -0.6895166, -1.37517223],
#                     [2.81742948, -0.01780752, 0.18363679],
#                     [2.20422291, -0.77223806, 0.20893524],
#                     [3.38891525, -0.17263024, -0.5686021],
#                     [-2.86669414, -0.14282213, -0.31653989],
#                     [-2.17356321, -0.01889467, -0.98894102],
#                     [-3.61843908, 0.36974668, -0.61083718]])
#     hex = pv.Constants.convert(hex, 'angstroms', to_AU=True)
#     water_coord = np.expand_dims(hex, 0)
#     start_cds = np.tile(water_coord, (20000, 1, 1))
#     impo = pv.ImpSampManager(trial_function='trial_wavefunction',
#                                   trial_directory=potDir,
#                                   python_file='call_trial.py',
#                                   pot_manager=harm_pot,
#                                   deriv_function='dpsi_dx')
#
#     myDMC = pv.DMC_Sim(sim_name="hex_impsamp_test_ana",
#                        output_folder=sim_ex_dir,
#                        weighting='discrete',
#                        num_walkers=20000,
#                        num_timesteps=100,
#                        equil_steps=5,
#                        chkpt_every=10,
#                        wfn_every=10,
#                        desc_wt_steps=5,
#                        atoms=["O", "H", "H"]*6,
#                        delta_t=1,
#                        potential=harm_pot,
#                        imp_samp=impo,
#                        log_every=1,
#                        start_structures=start_cds,
#                        )
#     myDMC.run()
