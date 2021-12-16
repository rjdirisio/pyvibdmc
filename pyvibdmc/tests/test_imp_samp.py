import pytest
import pyvibdmc as pv
import os
import numpy as np

sim_ex_dir = "imp_samp_results"


def test_run_dmc_short():
    import shutil
    if os.path.isdir(sim_ex_dir):
        shutil.rmtree(sim_ex_dir)

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager_NoMP(trial_function='trial_harm',
                                  trial_directory=potDir,
                                  python_file='harm_trial_wfn.py',
                                  deriv_function='derivative',
                                  chdir=True)

    myDMC = pv.DMC_Sim(sim_name="harm_osc_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=1000,
                       num_timesteps=500,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["O-H"],
                       delta_t=1,
                       potential=harm_pot,
                       second_impsamp_displacement=True,
                       imp_samp=impo,
                       imp_samp_oned=True,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       )
    myDMC.run()
    assert True


def test_run_dmc_short_morse():
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'morse_osc_1d.py'
    potFunc = 'oh_stretch_morse'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                             trial_directory=potDir,
                             python_file='harm_trial_wfn.py',
                             pot_manager=harm_pot,
                             deriv_function='derivative')

    myDMC = pv.DMC_Sim(sim_name="morse_osc_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=500,
                       num_timesteps=100,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["O-H"],
                       delta_t=1,
                       potential=harm_pot,
                       imp_samp=impo,
                       imp_samp_oned=True,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       )
    myDMC.run()
    assert True

def test_run_dmc_large_ts_morse():
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'morse_osc_1d.py'
    potFunc = 'oh_stretch_morse'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                             trial_directory=potDir,
                             python_file='harm_trial_wfn.py',
                             pot_manager=harm_pot,
                             deriv_function='derivative')

    myDMC = pv.DMC_Sim(sim_name="large_ts_impsamp",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=500,
                       num_timesteps=1000,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["O-H"],
                       delta_t=10,
                       potential=harm_pot,
                       imp_samp=impo,
                       imp_samp_oned=True,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       )
    myDMC.run()
    assert True


# def test_water():
#     # initialize potential
#     potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/FortPots/Partridge_Schwenke_H2O/')
#     # purposes
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
#     start_coord = np.expand_dims(water_coord, axis=0)  # Make it (1 x num_atoms x 3)
#
#     ex_args = {'dists':[[0,2],[2,1]],
#                'angs':[[0,2,1]]}
#     impo = pv.ImpSampManager(trial_function='trial_wavefunction',
#                              trial_directory=potDir,
#                              python_file='call_trl_h2o.py',
#                              pot_manager=harm_pot,
#                              deriv_kwargs=ex_args,
#                              trial_kwargs=ex_args)
#
#     myDMC = pv.DMC_Sim(sim_name="water_impsamp_test",
#                        output_folder=sim_ex_dir,
#                        weighting='discrete',
#                        num_walkers=2000,
#                        num_timesteps=5000,
#                        equil_steps=5,
#                        chkpt_every=10,
#                        wfn_every=10,
#                        desc_wt_steps=5,
#                        atoms=["H", "H", "O"],
#                        delta_t=1,
#                        potential=harm_pot,
#                        imp_samp=impo,
#                        log_every=1,
#                        start_structures=start_coord,
#                        )
#     myDMC.run()
#     assert True