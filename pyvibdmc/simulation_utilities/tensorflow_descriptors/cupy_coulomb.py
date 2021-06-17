import numpy as xp
try:
    import cupy as xp
except ImportError:
    print('Cupy not installed, defaulting to np')

import itertools as itt

__all__=['Coulomb_It']

class Coulomb_It:
    def __init__(self, zs, sort_mat=False, like_atoms=None):
        self.zs = xp.array(zs)
        self.sort_mat = sort_mat
        self.like_atoms = like_atoms
        self._initialize()

    def _initialize(self):
        self.num_atoms = len(self.zs)

        if self.like_atoms is not None and self.sort_mat:
            flat_like = [item for sublist in self.like_atoms for item in sublist]
            if len(flat_like) != self.num_atoms:
                raise ValueError("Please put all atoms in like_atoms list")

        self.idxs = list(itt.combinations(range(self.num_atoms), 2))
        self.idxs_0 = [z[0] for z in self.idxs]
        self.idxs_1 = [o[1] for o in self.idxs]

    def atm_atm_dists(self, cds, mat=True):
        """
        Takes in coordinates and a boolean that specifies whether or not this will return a vector or matrix.
        This fills the diagonal with 1!!!
        """
        ngeoms = cds.shape[0]
        dists = xp.zeros((ngeoms, len(self.idxs)))
        for idx_num, idx in enumerate(self.idxs):
            atm_0 = cds[:, idx[0]]
            atm_1 = cds[:, idx[1]]
            dist = xp.linalg.norm(atm_0 - atm_1, axis=1)
            dists[:, idx_num] = dist
        if mat:
            result = xp.zeros((ngeoms, self.num_atoms, self.num_atoms))
            result[:, self.idxs_0, self.idxs_1] = dists
            result[:, self.idxs_1, self.idxs_0] = dists
            # fill diagonal with 1s
            result[:, xp.arange(self.num_atoms), xp.arange(self.num_atoms)] = 1
            return result
        else:
            return dists

    def sort_coulomb(self, c_mat):
        num_geoms = c_mat.shape[0]
        """Takes in coulomb matrix and sorts it according to the row norm"""
        tot_inds = xp.zeros((num_geoms, len(c_mat[0])), dtype=int)
        for pairz in self.like_atoms:
            if len(pairz) == 1:
                tot_inds[:, pairz[0]] = xp.tile(pairz, num_geoms)
            else:
                pairz = xp.array(pairz)
                # argsort the pairs
                norm_mat = xp.argsort(-1 * xp.linalg.norm(c_mat, axis=1))
                relevant_idcs = xp.isin(norm_mat, pairz)
                pair_norm_mat = norm_mat[relevant_idcs].reshape(len(norm_mat), len(pairz))
                tot_inds[:, pairz] = pair_norm_mat
        # swap the pairs as they were
        c_mat = c_mat[xp.arange(c_mat.shape[0])[:, None, None], tot_inds[:, None, :], tot_inds[:, :, None]]
        return c_mat

    def run(self, cds):
        """
        Takes in cartesian coordinates, outputs the upper triangle of
        the coulomb matrix.  Option to have it sorted for permutational invariance
        """
        # get 0.5 * z^0.4
        rest = xp.ones((len(self.zs), len(self.zs)))
        xp.fill_diagonal(rest, 0.5 * self.zs ** 0.4)
        # get zii^2/zij matrix
        zij = xp.outer(self.zs, self.zs)
        # rij
        atm_atm_mat = self.atm_atm_dists(cds, mat=True)
        coulomb = zij * rest / atm_atm_mat
        # sort each according to norm of rows/columns
        if self.sort_mat:
            coulomb_s = self.sort_coulomb(coulomb)
        else:
            coulomb_s = coulomb
        upper_coulomb = coulomb_s[:, self.idxs_0, self.idxs_1]
        return upper_coulomb
