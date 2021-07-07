import pytest
import numpy as np
from pyvibdmc import DistIt

udu = np.array([[-5.1637122053302426E-002, -0.64283289924786391, 1.5013908299665704],
                [-6.6986645862392238E-002, -1.0712440718151974, 0.63206523297920003],
                [0.68029777946852377, -1.0421568612924319, 1.9715519867047862],
                [-6.2517576960482130E-002, -0.96998368494931764, -1.2761479660404149],
                [-0.10689894504689866, -3.4286135083244317E-003, -1.2309713446779025],
                [-0.74897513808594363, -1.2354360967793472, -1.8873593576677006],
                [2.5686843929439932E-002, 1.6025162419831362, -0.16918517749731218],
                [2.0116888927609659E-002, 1.0626137857410543, 0.63588396614107578],
                [-0.59312555997327554, 2.3143071935596669, -9.0920564389553662E-003]])
uuu = np.array([[-2.5959991041286115E-002, 1.4991768513805555, -0.60084982145893295],
                [0.55551975361128614, 2.2546995583418465, -0.67743590894778982],
                [3.1332763535393070E-002, 1.2106512728103587, 0.32154786324596829],
                [-2.5969936853924985E-002, -1.2699370467222246, -0.99790377379813366],
                [3.1321499633255201E-002, -0.32685620892671469, -1.2092399775892559],
                [0.55548007897032570, -1.7140334262004899, -1.6139337867371595],
                [-2.5940453410975263E-002, -0.22924002701551754, 1.5987489246721986],
                [0.55556956122359069, -0.54066260106652508, 2.2913245852660338],
                [3.1338180636427793E-002, -0.88380042635289757, 0.88768175980027386]])
cds = np.array([udu, uuu])


def test_dist():
    """Unsorted distance matrix descriptor"""
    transformer = DistIt(zs=[8, 1, 1] * 3,
                         method='distance',
                         full_mat=False)
    dist_vec = transformer.run(cds)
    transformer = DistIt(zs=[8, 1, 1] * 3,
                         method='distance',
                         full_mat=True)
    dist_mat = transformer.run(cds)
    assert True

def test_spf_unsorted():
    """Sorted/Unsorted distance matrix descriptor"""
    from pyvibdmc import DistIt
    transformer = DistIt(zs=[8, 1, 1] * 3,
                         method='spf',
                         eq_xyz=cds[0],
                         full_mat=False,
                         )
    dist_mat_sorted = transformer.run(cds)
    assert True

def test_spf():
    """Sorted/Unsorted distance matrix descriptor"""
    from pyvibdmc import DistIt
    transformer = DistIt(zs=[8, 1, 1] * 3,
                         method='spf',
                         eq_xyz=cds[0],
                         full_mat=False,
                         sorted_atoms=[[0], [1, 2], [3], [4, 5], [6], [7, 8]]
                         )
    dist_mat_sorted = transformer.run(cds)
    assert True


def test_coulomb():
    """Sorted/Unsorted distance matrix descriptor"""
    from pyvibdmc import DistIt
    transformer = DistIt(zs=[8, 1, 1] * 3,
                         method='coulomb',
                         full_mat=True,
                         sorted_atoms=[[0], [1, 2], [3], [4, 5], [6], [7, 8]],
                         sorted_groups=[[0, 1, 2], [6, 7, 8]]
                         )
    dist_mat_sorted = transformer.run(np.tile(udu, (10, 1, 1)))
    assert True


def test_sorting():
    # Swap water 1 and water 2
    w1 = [0, 1, 2]
    w2 = [3, 4, 5]
    tot_atms = np.arange(len(udu))
    tot_atms_c = np.copy(tot_atms)
    tot_atms_c[w1] = w2
    tot_atms_c[w2] = w1
    w2w1 = np.copy(udu[tot_atms_c])

    # Swap H1 and H2 on water 1
    cop = np.copy(udu)
    cop[[1, 2]] = cop[[2, 1]]
    w1mod = cop

    # Swap H1 and H2 in water 1 in w2w1
    cop = np.copy(w2w1)
    cop[[1, 2]] = cop[[2, 1]]
    w2w1_w1mod = cop

    # All should have the same sorted spf
    srt_cds = np.array([udu, w2w1, w1mod, w2w1_w1mod])

    from pyvibdmc import DistIt
    transformer = DistIt(zs=[8, 1, 1] * 3,
                         method='spf',
                         eq_xyz=cds[0],
                         full_mat=False,
                         sorted_groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8], ],
                         sorted_atoms=[[0], [1, 2], [3], [4, 5], [6], [7, 8]]
                         )
    dist_mat_sorted = transformer.run(srt_cds)
    assert np.count_nonzero(np.diff(dist_mat_sorted,axis=0)) == 0


def test_sorting_2():
    # Swap water 1 and water 2
    w1 = [0, 1, 2]
    w2 = [3, 4, 5]
    tot_atms = np.arange(len(uuu))
    tot_atms_c = np.copy(tot_atms)
    tot_atms_c[w1] = w2
    tot_atms_c[w2] = w1
    w2w1 = np.copy(uuu[tot_atms_c])

    # Swap H1 and H2 on water 1
    cop = np.copy(uuu)
    cop[[1, 2]] = cop[[2, 1]]
    w1mod = cop

    # Swap H1 and H2 in water 1 in w2w1
    cop = np.copy(w2w1)
    cop[[1, 2]] = cop[[2, 1]]
    w2w1_w1mod = cop

    # All should have the same sorted spf
    srt_cds = np.array([uuu, w2w1, w1mod, w2w1_w1mod])

    from pyvibdmc import DistIt
    transformer = DistIt(zs=[8, 1, 1] * 3,
                         method='spf',
                         eq_xyz=cds[0],
                         full_mat=False,
                         sorted_groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8], ],
                         sorted_atoms=[[0], [1, 2], [3], [4, 5], [6], [7, 8]]
                         )
    dist_mat_sorted = transformer.run(srt_cds)
    assert np.count_nonzero(np.diff(dist_mat_sorted,axis=0)) == 0