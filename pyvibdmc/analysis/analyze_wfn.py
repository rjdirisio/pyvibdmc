import numpy as np
import numpy.linalg as la


class AnalyzeWfn:
    def __init__(self, coordinates):
        """Takes xyz coordinates of an molecule, and calculates attributes associated with the molecule. This is built
        with DMC wave functions in mind, but can be used for any xyz atom. This class handles bond lengths, angles, etc.

        :param coordinates: mx3 array, or nxmx3 array, n = number of geometries, m = number of atoms, 3 is x y z
        :type coordinates: np.ndarray"""
        self.xx = coordinates
        if len(self.xx.shape) == 2:  # if only one geometry, make two
            self.xx = np.expand_dims(self.xx, axis=0)

    @staticmethod
    def dot_pdt(v1, v2):
        """Takes two stacks of vectors and dots the pairs of vectors together"""
        new_v1 = np.expand_dims(v1, axis=1)
        new_v2 = np.expand_dims(v2, axis=2)
        return np.matmul(new_v1, new_v2).squeeze()

    @staticmethod
    def exp_val(operator, dw):
        """
        Calculation of an expectation value using Monte Carlo Integration

        :param operator: array of bond lengths, bond angles, potentials, dipole moments, etc.
        :type operator: np.ndarray
        :param dw: Descendant Weights
        :type dw: np.ndarray
        :return: Expectation value (single float)
        """
        if len(operator.shape) > 1:
            return np.average(operator, axis=0, weights=dw)
        else:
            return np.average(operator, weights=dw)

    def bond_length(self, atm1, atm2):
        """
        Computes the norm of the vector between two atoms.

        :param atm1: desired atom number 1  (python index starts at 0)
        :type atm1: int
        :param atm2: desired atom number 2
        :type atm2: int
        :return: norm of x[atm1]-x[atm2]
        """
        return la.norm(self.xx[:, atm1] - self.xx[:, atm2], axis=1)

    def bond_angle(self, atm1, atm_vert, atm3):
        """Calculate atm1 - atm_vert - atm3 angle

        :param atm1: Index of the first external atom   (python index starts at 0)
        :type atm1: int
        :param atm_vert: Index of the atom at the vertex  (python index starts at 0)
        :type atm_vert: int
        :param atm3: Index of the second external atom   (python index starts at 0)
        :type atm3: int
        :return: Angle in radians
        """
        vec1 = self.xx[:, atm1] - self.xx[:, atm_vert]
        vec2 = self.xx[:, atm3] - self.xx[:, atm_vert]
        # cos(x) = left.right/(|left|*|right|)
        dotV = np.arccos(self.dot_pdt(vec1, vec2) /
                         (la.norm(vec1, axis=1) * la.norm(vec2, axis=1)))
        return dotV

    def bisecting_vector(self, atm1, atm_vert, atm3):
        """
        Get  normalized bisecting vector of two vectors (two bond lengths).
        It is assumed that the two flanking atoms have the same central atom.
        |b|*a + |a|*b
        
        :param atm1: Index of the first external atom   (python index starts at 0)
        :type atm1: int
        :param atm_vert: Index of the atom at the vertex  (python index starts at 0)
        :type atm_vert: int
        :param atm3: Index of the second external atom   (python index starts at 0)
        :type atm3: int
        :return: normalized bisecting vector
        """
        b1 = self.xx[:, atm1] - self.xx[:, atm_vert]
        b2 = self.xx[:, atm3] - self.xx[:, atm_vert]
        bis = la.norm(b1, axis=1) * b2 + la.norm(b2, axis=1) * b1
        return bis / la.norm(bis, axis=1)

    def dihedral(self, vec1, vec2, vec3):
        """Looking down vec2, calculate dihedral angle
        https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
        https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
        """
        crossterm1 = np.cross(np.cross(vec1, vec2, axis=1),
                              np.cross(vec2, vec3, axis=1), axis=1)

        term1 = self.dot_pdt(crossterm1, (vec2 / la.norm(vec2, axis=1)[:, np.newaxis]))

        term2 = self.dot_pdt(np.cross(vec1, vec2, axis=1), np.cross(vec2, vec3, axis=1))

        dh = np.arctan2(term1, term2)
        return dh

    def get_component(self, atm, xyz):
        """Get x, y, or z component of a vector that corresponds to a particular atom in some predetermined cooridnate
        system.

        :param atm: atom's index in numpy array
        :type: int
        :param xyz: Either 0 (x), 1 (y), or 2 (z)
        :type xyz: int
        :return: vector of x, y, or z components of a stack of vectors
        """
        return self.xx[:, atm, xyz]

    @staticmethod
    def cart_to_spherical(vecc):
        """
        Takes stack of cartesian vectors (nx3) and translates them to stack of r theta phi coordinates,
        with the assumption that you are already in the coordinate frame you desire.
        """
        r = la.norm(vecc, axis=1)
        th = np.arccos(vecc[:, -1] / r)  # z/r
        phi = np.arctan2(vecc[:, 1], vecc[:, 0])  # y/x
        return r, th, phi

    @staticmethod
    def projection_1d(attr, desc_weights, bin_num=25, range=None, normalize=True):
        """
        Project the probability amplitude onto a particular coordinate, 1D Histogram.
        
        :param attr: the coordinate to be projected onto (bond length, bond angle, etc.)
        :type attr: np.ndarray
        :param desc_weights: the descendant weights for \psi**2
        :type desc_weights: np.ndarray
        :param bin_num: Number of bins in histogram (higher number, more number of walkers needed)
        :type attr: int
        :param range: Range over which attr is histogrammed
        :type range: tuple
        :param normalize: Normalizes the wave function along coordinate
        :type attr: bool
        :return: np.ndarray of shape (bin_num-1 x 2), bin centers, amplitude at bin centers
        """
        amp, bin_edge = np.histogram(attr,
                                    bins=bin_num,
                                    range=range,
                                    weights=desc_weights,
                                    density=normalize)
        xx = 0.5 * (bin_edge[1:] + bin_edge[:-1])  # get bin centers
        return np.column_stack((xx, amp))

    @staticmethod
    def projection_2d(attr1, attr2, desc_weights, bin_num=[25, 25], range=None, normalize=True):
        """
        Project the probability amplitude onto a 2 coordinates, 2D Histogram.

        :param attr1: coordinate 1 to be projected onto (bond length, bond angle, etc.)
        :type attr: np.ndarray
        :param attr2: coordinate 2 to be projected onto (bond length, bond angle, etc.)
        :type attr: np.ndarray
        :param desc_weights: the descendant weights for \psi**2
        :type desc_weights: np.ndarray
        :param bin_num: Number of bins for histogram in both directions (higher number, more number of walkers needed)
        :type attr: tuple
        :param range: Range over which attr is histogrammed
        :type range: list of two tuples
        :param normalize: Normalizes the wave function along both coordinates (integrated area under 2D surface is 1)
        :type attr: bool
        :return: np.ndarray of shape (bin_num-1 x 2), bin centers x, bin centers y, and then the 2D array with amplitude.
        """
        amps, bin_edges_x, bin_edges_y = np.histogram2d(attr1,
                                                        attr2,
                                                        bins=bin_num,
                                                        range=range,
                                                        weights=desc_weights,
                                                        density=normalize)
        amps = amps.T  # to get x and y to match up with what is expected
        bins_x = 0.5 * (bin_edges_x[1:] + bin_edges_x[:-1])
        bins_y = 0.5 * (bin_edges_y[1:] + bin_edges_y[:-1])
        return bins_x, bins_y, amps
