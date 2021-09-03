import numpy as np

__all__ = ['MolFiniteDifference']


class MolFiniteDifference:
    """Helper to calculate derivatives of some value as a function of Cartesian displacements in a molecule."""
    weights_der1 = {'3': np.array([-1 / 2, 0, 1 / 2]),
                    '5': np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
                    }
    weights_der2 = {'3': np.array([1, -2, 1]),
                    '5': np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
                    }

    @staticmethod
    def displace_molecule(eq_geom, atm_cd, dx, num_disps):
        """
        Displace atm along cd
        :param eq_geom: Geometry from which you will be displaced
        :param atm_cd: a tuple that has a paricular atom of interest to displace in a particular dimension (x,y,or,z)
        :param dx: The amount each geometry will be replaced
        :param num_disps: int of how many displacements to do, will take the form of 3 or 5 for harmonic analysis.
        :return: Displaced coordinates in a 3D array (n,m,3). If displaced in two directions, then still (n,m,3)
        """
        in_either_direction = num_disps // 2
        dx_ordering = np.arange(-in_either_direction, in_either_direction + 1)
        dx_ordering = dx_ordering * dx
        if len(atm_cd) == 2:
            """Displace one atom in each direction"""
            atm = atm_cd[0]  # atom of interest
            cd = atm_cd[1]  # x,y, or z
            displaced_cds = np.zeros(np.concatenate(([len(dx_ordering)], np.shape(eq_geom))))
            for disp_num, disp in enumerate(dx_ordering):
                dx_atms = np.zeros(eq_geom.shape)
                dx_atms[atm, cd] += disp
                displaced_cds[disp_num] = eq_geom + dx_atms
        elif len(atm_cd) == 4:
            """Displace two atoms in each direction. For 2D (mixed) derivatives"""
            atm1 = atm_cd[0]
            cd1 = atm_cd[1]
            atm2 = atm_cd[2]
            cd2 = atm_cd[3]
            displaced_cds = np.zeros(np.concatenate(([len(dx_ordering) ** 2], np.shape(eq_geom))))
            ct = 0
            for disp_num, disp in enumerate(dx_ordering):
                for disp_num_2, disp2 in enumerate(dx_ordering):
                    dx_atms = np.zeros(eq_geom.shape)
                    dx_atms[atm1, cd1] += disp
                    dx_atms[atm2, cd2] += disp2
                    displaced_cds[ct] = eq_geom + dx_atms
                    ct += 1
        return displaced_cds

    @classmethod
    def differentiate(cls, values, dx, num_points, der):
        if der == 1:  # First derivative, one dimension
            wts = cls.weights_der1[str(num_points)]
            wts = wts / dx
            diff = np.dot(wts, values)
        elif der == 11:  # Mixed first derivative
            wts = cls.weights_der1[str(num_points)]
            wts = wts / dx
            diff = np.dot(wts, np.dot(wts, values))
        elif der == 2:  # Second derivative, one dimension
            wts = cls.weights_der2[str(num_points)]
            wts = wts / dx ** 2  # since it's one dimensional
            diff = np.dot(wts, values)
        return diff
