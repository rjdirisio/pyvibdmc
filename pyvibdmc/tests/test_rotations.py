from ..analysis import *
import pytest


# Test rotating coords to xy plane
def test_generate_xyz_rot():
    ch5 = np.array([[3.64370000e-05, 2.61250000e-04, 2.05490000e-05],
                    [6.96823202e-02, 1.10553262e+00, 2.26690000e-05],
                    [9.45749638e-01, -7.32895080e-01, -6.79440001e-05],
                    [1.18207493e+00, 1.87991209e-01, 9.14499999e-06],
                    [-4.36525904e-01, -3.33258609e-01, -9.39243426e-01],
                    [-4.36588351e-01, -3.33182565e-01, 9.39259010e-01]])
    ch5_stack = np.tile(ch5, (10, 1, 1))
    new_geoms = MolRotator.rotate_to_xy_plane(ch5_stack, orig=0, xax=1, xyp=4)
    test_zeros = np.concatenate((new_geoms[0, 0], new_geoms[0, 1, 1:], [new_geoms[0, 4, -1]]))
    all_close = np.allclose(test_zeros, np.zeros(6))
    assert all_close
