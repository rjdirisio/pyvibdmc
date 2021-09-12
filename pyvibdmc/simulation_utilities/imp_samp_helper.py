import numpy as np


def dr_dx(cds, atm_bonds):
    """
    calculates the dr/dx first derivatives for bond lengths
    :param cds: num_walkers x num_atms x 3 array
    :param atm_bonds: list of lists of bond lengths that are relevant to the trial wave function. like [[0,1],[0,2]]
    :param d_ar: Optional. The derivative array that can be built up starting with this one or in another function
    :return: len(atm_bonds) x num_walkers x num_atoms x 3 array of dr/dx
    """
    d_ar = np.zeros((len(atm_bonds),) + cds.shape)  # the first derivative array
    for num, atm_pair in enumerate(atm_bonds):
        diff_carts = cds[:, atm_pair[0]] - cds[:, atm_pair[1]]  # alpha values
        r = np.linalg.norm(diff_carts, axis=1)
        d_ar[num, :, atm_pair[0]] = diff_carts / r[:, np.newaxis]
        d_ar[num, :, atm_pair[1]] = -1 * diff_carts / r[:, np.newaxis]
    return d_ar


def dth_dx(cds, atm_bonds):
    """
        calculates the dtheta/dx first derivatives for bond lengths
        :param cds: num_walkers x num_atms x 3 array
        :param atm_bonds: list of lists of bond lengths that are relevant to the trial wave function. like [[0,2,1]]
        :param d_ar: Optional. The derivative array that can be built up starting with this one or in another function
        :return: len(atm_bonds) x num_walkers x num_atoms x 3 array of dr/dx
        """
    pass

if __name__ == '__main__':
    water_coord = np.array([[1.81005599, 0., 0.],
                            [-0.45344658, 1.75233806, 0.],
                            [0., 0., 0.]])
    water_coord = np.array([water_coord, water_coord])
    xx = dr_dx(water_coord, [[0, 2], [1, 2]])
    print('hi')
