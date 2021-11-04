import pyvibdmc as pv
import pytest
import numpy as np
import sys,os

sim_ex_dir = "adiabatic_results"

def test_run_adiabatic():
    import shutil
    if os.path.isdir(sim_ex_dir):
        shutil.rmtree(sim_ex_dir)

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm_shifted'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)
    def calc_x(cds):
        return np.squeeze(cds)


    myDMC = pv.DMC_Sim(sim_name="harm_osc_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=50,
                       num_timesteps=200,
                       equil_steps=100,
                       chkpt_every=50,
                       wfn_every=200,
                       desc_wt_steps=199,
                       atoms=["O-H"],
                       delta_t=1,
                       potential=harm_pot,
                       log_every=50,
                       start_structures=np.reshape(pv.Constants.convert(0.98,'angstroms',to_AU=True),(1,1,1)),
                       cur_timestep=0,
                       adiabatic_dmc={'initial_lambda': 0,
                                    'lambda_change': 1e-6,
                                    'equil_time': 20,
                                    'observable_func': calc_x}
                       )
    myDMC.run()
    assert True

def test_analyze_adiabatic():
    import matplotlib.pyplot as plt
    sim = pv.SimInfo(f'{sim_ex_dir}/harm_osc_test_sim_info.hdf5')
    vref = sim.get_vref(ret_cm=True)
    pv.Plotter.plt_vref_vs_tau(vref,f'{sim_ex_dir}/admc_vref.png')
    lam = np.load(f'{sim_ex_dir}/harm_osc_test_lambda.npy')
    lam = pv.Constants.convert(lam, 'wavenumbers', to_AU=False)
    lam = pv.Constants.convert(lam, 'angstroms', to_AU=True)
    # lam = pv.Constants.convert(lam, 'angstroms', to_AU=True)
    plt.plot(lam,vref[:,1],'k')
    coefs = np.polyfit(lam[20:], vref[20:,1], 3)
    print(coefs[-2:])
    plt.plot(lam[20:], np.polyval(coefs[-2:], lam[20:]))
    plt.xlabel('$\lambda$')
    plt.ylabel('$W(\lambda)$')
    # plt.show()