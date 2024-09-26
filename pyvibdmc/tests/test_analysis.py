import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys

import pyvibdmc as pv

test_sim = pv.SimInfo('pyvibdmc/sample_sim_data/tutorial_water_0_sim_info.hdf5')
savefigpth = 'pyvibdmc/tests/'


# Test writing and reading xyz coords from file
def test_write_xyz_file():
    cds, dws = test_sim.get_wfns([2500, 3500],ret_ang=True)  # get two wave functions just for testing
    atm_str_list = ["H", "H", "O"]
    pv.XYZNPY.write_xyz(coords=cds, fname=f'{savefigpth}water_cds.xyz', atm_strings=atm_str_list,
                        cmt='from dmc simulation')
    cds_back = pv.XYZNPY.extract_xyz(f'{savefigpth}water_cds.xyz', num_atoms=len(atm_str_list))
    assert np.allclose(cds, cds_back)


def test_get_all_siminfo():
    vref_vs_tau = test_sim.get_vref()
    pop_vs_tau = test_sim.get_pop()
    atom_nums = test_sim.get_atomic_nums()
    atom_masses = test_sim.get_atom_masses()


# Basic SimInfoStuff
def test_sim_data_zpe():
    zpe = test_sim.get_zpe(onwards=1000, ret_cm=True)


def test_zpe_std():
    zpes = []
    for sim_num in range(5):
        test_sim = pv.SimInfo(
            f'pyvibdmc/sample_sim_data/tutorial_water_{sim_num}_sim_info.hdf5')  # 5 independent DMC sims!
        this_zpe = test_sim.get_zpe(onwards=100, ret_cm=True)
        zpes.append(this_zpe)
    final_zpe = np.average(zpes)
    final_std_dev = np.std(zpes)


def test_sim_combine_wfns():
    combined_wfns, dws = test_sim.get_wfns([2500, 3500], ret_ang=True)  # the time steps you want to include


# Internal Analyzer Test
def test_ana_wfn_blens():
    """
    Test just the wave function analyzer
    """
    expected_blens = np.array([1.81005599, 1.81505599, 1.82005599, 1.82505599, 1.83005599, 1.83505599,
                               1.84005599, 1.84505599, 1.85005599, 1.85505599])
    water = np.array([[1.81005599, 0., 0.],
                      [-0.45344658, 1.75233806, 0.],
                      [0., 0., 0.]])
    sample_waters = np.broadcast_to(water, (10,) + np.shape(water)).copy()  # make ten copies

    for wn, water in enumerate(sample_waters):
        water[0, 0] += 0.005 * wn

    ana_o = pv.AnalyzeWfn(sample_waters)
    bond_lengths = ana_o.bond_length(0, 2)  # OH bond length
    print(bond_lengths)
    assert np.allclose(bond_lengths, expected_blens)


def test_dihedral():
    # CHHO - formaldehyde has an improper dihedral angle.
    formie = np.array([[0.0, 0.538978, 0.000000],
                       [0.05, 1.122055, 0.946279],
                       [0.0, 1.122055, -0.946279],
                       [0.0, -0.684747, 0.0]])
    # This is an equilibrium configuration of the water dimer
    dimie = np.array([[1.513632, -0.005249, -0.121857],
                      [0.560102, 0.002812, 0.048059],
                      [1.913196, 0.033035, 0.750687],
                      [-1.385643, 0.004325, 0.110302],
                      [-1.750594, 0.746224, -0.382028],
                      [-1.746613, -0.774680, -0.324277]])
    formie = np.tile(formie, (2, 1, 1))
    dimie = np.tile(dimie, (2, 1, 1))
    analyzer_dim = pv.AnalyzeWfn(dimie)
    # dim_nums = [3,1,4,5]
    angle = np.degrees(analyzer_dim.dihedral(3 - 1, 1 - 1, 4 - 1, 5 - 1))
    assert pytest.approx(angle, abs=1e-1) == 122.6

    angle2 = np.degrees(analyzer_dim.dihedral(3 - 1, 1 - 1, 4 - 1, 6 - 1))
    assert pytest.approx(angle2, abs=1e-1) == -123.1

    # Improper Dihedral
    analyzer_form = pv.AnalyzeWfn(formie)

    # form_nums = [3,1,4,2]
    improp = np.degrees(analyzer_form.dihedral(3 - 1, 1 - 1, 4 - 1, 2 - 1))
    assert pytest.approx(improp, abs=1) == 177
    improp2 = np.degrees(analyzer_form.dihedral(2 - 1, 1 - 1, 4 - 1, 3 - 1))
    assert pytest.approx(improp2, abs=1) ==-177


# Plotting and analyzing tests
def test_plt_atm_atm_dists():
    import itertools as itt
    cds, dws = test_sim.get_wfns(np.arange(1500, 5000, 1000), ret_ang=True)  # get two wave functions just for testing
    analyzer = pv.AnalyzeWfn(cds)  # initialize analyzer object

    num_atoms = cds.shape[1]
    combos = itt.combinations(range(num_atoms), 2)
    for combo in combos:  # for each pair of atom-atom distances, calculate the bond length for each walker
        cur_bl = analyzer.bond_length(combo[0], combo[1])

        bl_histo = analyzer.projection_1d(attr=cur_bl,  # make a 1d histogram , x/y data
                                          desc_weights=dws,
                                          range=(0, 3))

        pv.Plotter.plt_hist1d(hist=bl_histo,  # plot histogram x/y data
                              xlabel=f"Bond Length R{combo[0]}-R{combo[1]} (Angstroms)",
                              save_name=f'{savefigpth}BondLength_R{combo[0]}R{combo[1]}.png')


def test_plt_vref():
    pv.Plotter.plt_vref_vs_tau(vref_vs_tau=test_sim.get_vref(ret_cm=True),
                               save_name=f'{savefigpth}test_vref.png')
    assert True


def test_plt_water_angle():
    cds, dws = test_sim.get_wfns(np.arange(1500, 5000, 1000),ret_ang=True)  # get two wave functions just for testing

    analyzer = pv.AnalyzeWfn(cds)  # initialize analyzer object
    hoh_angle = analyzer.bond_angle(atm1=0,
                                    atm_vert=2,
                                    atm3=1)  # [H H O], so atm[2] at vertex

    hoh_angle = np.rad2deg(hoh_angle)  # analyzer returns in radians

    hoh_histo = analyzer.projection_1d(attr=hoh_angle,  # make a 1d histogram , x/y data
                                       desc_weights=dws,
                                       bins=20,
                                       range=(60, 150))

    pv.Plotter.plt_hist1d(hist=hoh_histo,  # plot histogram x/y data
                          xlabel=r"HOH Angle $\rm{\theta}$ (Degrees)",
                          save_name=f'{savefigpth}HOH_angle.png')


def test_adv_plt_many_vrefs():
    """Plot lots of vrefs on top of each other"""

    # My favorite plotting settings
    params = {'text.usetex': False,
              'mathtext.fontset': 'dejavusans',
              'font.size': 14}
    plt.rcParams.update(params)

    # Time to plot!
    fig, ax = plt.subplots()
    for sim_num in range(5):
        temp_sim = pv.SimInfo(
            f'pyvibdmc/sample_sim_data/tutorial_water_{sim_num}_sim_info.hdf5')  # 5 independent DMC sims!
        this_vref = temp_sim.get_vref(ret_cm=True)
        ax.plot(this_vref[:, 0], this_vref[:, 1])
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel(r"Vref ($\rm{cm^{-1}}$)")
    fig.savefig(f"{savefigpth}ManyVrefs.png", dpi=300, bbox_inches='tight')
    plt.close()


def test_adv_plt_2dhistogram():
    # Advanced: Combine wave functions from across 5 simulations
    tot_cds = []
    tot_dw = []
    for sim_num in range(5):
        temp_sim = pv.SimInfo(
            f'pyvibdmc/sample_sim_data/tutorial_water_{sim_num}_sim_info.hdf5')  # 5 independent DMC sims!
        cds, dw = temp_sim.get_wfns(np.arange(1500, 5000, 1000),ret_ang=True)
        tot_cds.append(cds)
        tot_dw.append(dw)
    tot_cds = np.concatenate(tot_cds, axis=0)
    tot_dw = np.concatenate(tot_dw)

    # Advanced: 2D Histogram
    analyzer = pv.AnalyzeWfn(tot_cds)  # initialize analyzer object
    bond_len_OH1 = analyzer.bond_length(0, 2)
    bond_len_OH2 = analyzer.bond_length(1, 2)
    # bond_angle = np.rad2deg(analyzer.bond_angle(atm1=0,atm_vert=2,atm3=1))

    bins_x, bins_y, amp = analyzer.projection_2d(bond_len_OH1,
                                                 bond_len_OH2,
                                                 desc_weights=tot_dw,
                                                 bins=[15, 15],
                                                 range=[[0.5, 1.5], [0.5, 1.5]],
                                                 normalize=True)
    pv.Plotter.plt_hist2d(binsx=bins_x,
                          binsy=bins_y,
                          hist_2d=amp,
                          xlabel="ROH 1 (Angstroms)",
                          ylabel="ROH 2 (Angstroms)",
                          save_name=f"{savefigpth}2d_histogram.png")

# def test_parent_wts():
#     test_sim = pv.SimInfo('exSimResults/harm_osc_test_continuous_sim_info.hdf5')
#     cdz, wfns, parents = test_sim.get_wfns([100,200,300],ret_ang=True, get_parent_wts=True)
#     print('hi')