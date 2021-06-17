import itertools as itt

import numpy as xp
try:
    import cupy as xp
except ImportError:
    print('Cupy not installed, defaulting to np')

__all__=['DistIt']

class DistIt:
    """
    Calculates the Distance matrix for a given molecule. Returns a cupy or numpy multidimensional array.
    If cupy is installed, this will default to using it. This means in the run(cds) function, you must pass cds as a
    cupy array
    """

    def __init__(self,
                 zs,
                 sort_mat=False,
                 like_atoms=None,
                 full_mat=False):
        """
        :param zs: The list nuclear charges (Z) of each atom. If not sorting, this just gives us the number of atoms. Matters more for coulomb matrix.
        :type zs:list
        :param sort_mat: Boolean for wheter or not to sort the distance matrix.
        :type sort_mat:bool
        :param like_atoms: A list of lists that sorts the atoms in each sublist. MUST contain all atoms in the list of lists.
        :type like_atoms: list of lists
        :param full_mat: Whether to return the full matrix or the upper triangle, defaults to upper triangle
        :type full_mat: bool
        """
        self.zs = xp.array(zs)
        self.sort_mat = sort_mat
        self.like_atoms = like_atoms
        self.full_mat = full_mat
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

    def dist_matrix(self, atm_vec):
        """
        If full matrix required, calculate it here.
        This fills the diagonal with 1!!! (For coulomb matrix reasons)
        """
        ngeoms = atm_vec.shape[0]
        result = xp.zeros((ngeoms, self.num_atoms, self.num_atoms))
        result[:, self.idxs_0, self.idxs_1] = atm_vec
        result[:, self.idxs_1, self.idxs_0] = atm_vec
        # fill diagonal with 1s
        result[:, xp.arange(self.num_atoms), xp.arange(self.num_atoms)] = 1
        return result

    def atm_atm_dists(self, cds):
        """
        Takes in coordinates, will return a vector of the atm-atm dists.
        """
        ngeoms = cds.shape[0]
        dists = xp.zeros((ngeoms, len(self.idxs)))
        for idx_num, idx in enumerate(self.idxs):
            atm_0 = cds[:, idx[0]]
            atm_1 = cds[:, idx[1]]
            dist = xp.linalg.norm(atm_0 - atm_1, axis=1)
            dists[:, idx_num] = dist
        return dists

    def sort_dist(self, d_mat):
        num_geoms = d_mat.shape[0]
        """Takes in coulomb matrix and sorts it according to the row norm"""
        tot_inds = xp.zeros((num_geoms, len(d_mat[0])), dtype=int)
        for pairz in self.like_atoms:
            if len(pairz) == 1:
                tot_inds[:, pairz[0]] = xp.tile(pairz, num_geoms)
            else:
                pairz = xp.array(pairz)
                # argsort the pairs
                norm_mat = xp.argsort(-1 * xp.linalg.norm(d_mat, axis=1))
                relevant_idcs = xp.isin(norm_mat, pairz)
                pair_norm_mat = norm_mat[relevant_idcs].reshape(len(norm_mat), len(pairz))
                tot_inds[:, pairz] = pair_norm_mat
        # swap the pairs as they were
        d_mat = d_mat[xp.arange(d_mat.shape[0])[:, None, None], tot_inds[:, None, :], tot_inds[:, :, None]]
        return d_mat

    def run(self, cds):
        """
        Takes in cartesian coordinates, outputs the upper triangle of
        the coulomb matrix.  Option to have it sorted for permutational invariance
        """
        if len(cds.shape) == 2:
            cds = xp.array([cds, cds])
            ret_one = True
        else:
            ret_one = False
        # get zii^2/zij matrix
        # rij
        atm_atm_vec = self.atm_atm_dists(cds)
        if self.sort_mat or self.full_mat:
            atm_atm_mat = self.dist_matrix(atm_atm_vec)
        # sort each according to norm of rows/columns
        if self.sort_mat:
            dist_s = self.sort_dist(atm_atm_mat)
        else:
            dist_s = atm_atm_mat
        if not self.full_mat and self.sort_mat:  # if sorted but only want triangle
            dist_s = dist_s[:, self.idxs_0, self.idxs_1]

        if ret_one:
            return dist_s[0]
        else:
            return dist_s

#
# class SpfIT(DistIt):
#     """
#     Calculates the SPF matrix (r-r_e / r) for a given molecule. Returns a cupy or numpy multidimensional array.
#     """
#     def __init__(self,
#                  zs,
#                  eq_xyz,
#                  sort_mat=False,
#                  like_atoms=None,
#                  full_mat=False):
#         super().__init__(zs,sort_mat,like_atoms,full_mat)
#         self.eq_geom = eq_xyz
#         self.sort_mat = sort_mat
#         self.like_atoms = like_atoms
#         self.full_mat = full_mat
#         self._init_eq()
#
#     def _init_eq(self):
#         # Eq geom needs to be transformed into distance matrix
#         self.eq_geom = ([self.eq_geom,self.eq_geom])
#         r_eq = self.atm_atm_dists(self.eq_geom)
#         if self.sort_mat or self.full_mat:
#             r_eq = self.dist_matrix(r_eq)
#         return r_eq
#
#     def run(self, cds):
#         """
#         Takes in cartesian coordinates, outputs the upper triangle of
#         the coulomb matrix.  Option to have it sorted for permutational invariance
#         """
#         if len(cds.shape) == 2:
#             cds = xp.array([cds, cds])
#             ret_one = True
#         else:
#             ret_one = False
#         # get zii^2/zij matrix
#         # rij
#         atm_atm_vec = self.atm_atm_dists(cds)
#         if self.sort_mat or self.full_mat:
#             atm_atm_mat = self.dist_matrix(atm_atm_vec)
#         # sort each according to norm of rows/columns
#         if self.sort_mat:
#             dist_s = self.sort_dist(atm_atm_mat)
#         else:
#             dist_s = atm_atm_mat
#         if not self.full_mat and self.sort_mat:  # if sorted but only want triangle
#             dist_s = dist_s[:, self.idxs_0, self.idxs_1]
#
#         if ret_one:
#             return dist_s[0]
#         else:
#             return dist_s