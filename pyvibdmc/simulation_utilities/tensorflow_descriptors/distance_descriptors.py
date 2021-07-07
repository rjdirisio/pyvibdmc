import itertools as itt

__all__ = ['DistIt']


class DistIt:
    """
    Calculates the Distance matrix, coulomb matrix, or SPF matrix (delta_r / r) for a given molecule. Returns a cupy or
    numpy multidimensional array. If cupy is installed, this will default to using it.
    This means in the run(cds) function, you must pass cds as a cupy array
    """

    def __init__(self,
                 zs,
                 method,
                 eq_xyz=None,
                 sorted_groups=None,
                 sorted_atoms=None,
                 full_mat=False,
                 force_numpy=False):
        """
        @param zs: list of Nuclear charges of each of the atoms
        @param method: 'spf', 'distance', or 'coulomb'
        @param eq_xyz: Equilibrium structure from which the spf descriptor is constructed
        @param sorted_groups: a list of lists that only contains the atom groups one wants to swap.
        @param sorted_atoms: a list of lists containing ALL atoms. Will sort each sublist with themselves.
        @param full_mat: Return the full matrix rather than just the upper triangle. Upper triangle doesn't include diagonal.
        """
        self.zs = zs
        self.method = method.lower()
        self.eq_xyz = eq_xyz
        self.sorted_groups = sorted_groups
        self.sorted_atoms = sorted_atoms
        self.full_mat = full_mat
        self.force_numpy = force_numpy
        self._initialize()

    def check_cupy(self):
        import numpy as xp
        if not self.force_numpy:
            try:
                import cupy as xp
            except ImportError:
                print('Cupy not installed, defaulting to numpy')
        return xp

    def _initialize(self):
        self.xp = self.check_cupy()
        self.zs = self.xp.asarray(self.zs)
        self.num_atoms = len(self.zs)

        if self.sorted_atoms is not None or self.sorted_groups is not None:
            self.sort_mat = True
        else:
            self.sort_mat = False

        # Just check if all atoms are included in the sorted things
        if self.sorted_atoms is not None:
            flat_like = self.xp.concatenate(self.xp.asarray(self.sorted_atoms)).ravel()
            if len(flat_like) != self.num_atoms:
                raise ValueError("Please put all atoms in sorted_atoms list")

        if self.sorted_groups is not None:
            self.sorted_groups = self.xp.asarray(self.sorted_groups)

        self.idxs = list(itt.combinations(range(self.num_atoms), 2))
        self.idxs_0 = [z[0] for z in self.idxs]
        self.idxs_1 = [o[1] for o in self.idxs]

        if self.eq_xyz is None and self.method == 'spf':
            raise ValueError("eq_xyz is not set but using spf. Fix!")

        if self.eq_xyz is not None and self.method == 'spf':
            # Get eq value's dist mat
            self.eq_xyz = self.xp.asarray([self.eq_xyz, self.eq_xyz])
            prepped_r_eq = self.atm_atm_dists(self.eq_xyz)
            if not self.sort_mat:
                # This is the r_eq for all values if not sorted
                self.r_eq = prepped_r_eq[0]
            else:
                # Sort r_eq like you will the rest of the atoms
                prepped_r_eq = self.dist_matrix(prepped_r_eq)
                if self.sorted_atoms is not None:
                    prepped_r_eq = self.sort_atoms(prepped_r_eq)
                if self.sorted_groups is not None:
                    prepped_r_eq = self.sort_groups(prepped_r_eq)
                self.r_eq = prepped_r_eq[0]

    def dist_matrix(self, atm_vec):
        """
        If full matrix required, calculate it here.
        """
        ngeoms = atm_vec.shape[0]
        result = self.xp.zeros((ngeoms, self.num_atoms, self.num_atoms))
        result[:, self.idxs_0, self.idxs_1] = atm_vec
        result[:, self.idxs_1, self.idxs_0] = atm_vec
        # fill diagonal with appropriate coulomb matrix
        if self.method == 'coulomb':
            result[:, self.xp.arange(self.num_atoms), self.xp.arange(self.num_atoms)] = self.diag_coulomb
        return result

    def atm_atm_dists(self, cds):
        """
        Takes in coordinates, will return a vector of the atm-atm dists.
        """
        ngeoms = cds.shape[0]
        dists = self.xp.zeros((ngeoms, len(self.idxs)))
        for idx_num, idx in enumerate(self.idxs):
            atm_0 = cds[:, idx[0]]
            atm_1 = cds[:, idx[1]]
            dist = self.xp.linalg.norm(atm_0 - atm_1, axis=1)
            dists[:, idx_num] = dist
        return dists

    def sort_atoms(self, d_mat):
        num_geoms = d_mat.shape[0]
        """Takes in coulomb matrix and sorts it according to the row norm"""
        tot_inds = self.xp.zeros((num_geoms, len(d_mat[0])), dtype=int)
        for pairz in self.sorted_atoms:
            if len(pairz) == 1:
                tot_inds[:, pairz[0]] = self.xp.tile(pairz, num_geoms)
            else:
                pairz = self.xp.asarray(pairz)
                # argsort the pairs
                norm_mat = self.xp.argsort(-1 * self.xp.linalg.norm(d_mat, axis=1))
                relevant_idcs = self.xp.isin(norm_mat, pairz)
                pair_norm_mat = norm_mat[relevant_idcs].reshape(len(norm_mat), len(pairz))
                tot_inds[:, pairz] = pair_norm_mat
        # swap the pairs as they were
        d_mat = d_mat[self.xp.arange(num_geoms)[:, None, None], tot_inds[:, None, :], tot_inds[:, :, None]]
        return d_mat

    def sort_groups(self, d_mat):
        """
        Swap 2 or more groups of atoms according to the sum of the norm of the group's columns.
        Only one type of group can swapped.
        """
        num_geoms = d_mat.shape[0]
        num_pairz = len(self.sorted_groups)
        tot_inds = self.xp.arange(len(d_mat[0]), dtype=int)
        tot_inds = self.xp.tile(tot_inds, (num_geoms, 1))
        tot_normz = []
        for pairz in self.sorted_groups:
            pairz = self.xp.asarray(pairz, dtype=int)
            tot_norm = self.xp.sum(self.xp.linalg.norm(d_mat[:, :, pairz], axis=1), axis=1)
            tot_normz.append(tot_norm)
        tot_normz = self.xp.asarray(tot_normz).T
        sorted_gps = self.xp.argsort(-1 * tot_normz, axis=1)
        new_groups = self.sorted_groups[sorted_gps].reshape((-1, num_pairz * len(pairz)))
        tot_inds[:, self.sorted_groups.ravel()] = new_groups
        d_mat = d_mat[self.xp.arange(num_geoms)[:, None, None], tot_inds[:, None, :], tot_inds[:, :, None]]
        return d_mat

    def get_prepped_vec(self, atm_atm_vec):
        # Prepare matrices for sorting if needed. If not, we will just return these vectors as they are
        if self.method == 'coulomb':
            # get 0.5 * z^0.4
            rest = self.xp.ones((len(self.zs), len(self.zs)))
            self.xp.fill_diagonal(rest, 0.5 * self.zs ** 0.4)
            # get zii^2/zij matrix
            zij = self.xp.outer(self.zs, self.zs)
            skeleton = zij * rest
            self.diag_coulomb = self.xp.diag(skeleton)
            skeleton = skeleton[self.idxs_0, self.idxs_1]
            prepped_vec = self.xp.tile(skeleton, (len(atm_atm_vec), 1)) / atm_atm_vec
        else:
            prepped_vec = atm_atm_vec
        return prepped_vec
        # elif self.method == 'spf':
        #     prepped_vec = (atm_atm_vec - self.r_eq) / atm_atm_vec
        #
        # elif self.method == 'distance':
        #     prepped_vec = atm_atm_vec
        #
        # return prepped_vec

    def run(self, cds):
        """
        Takes in cartesian coordinates, outputs the descriptor matrix in either
        vector form (upper triangle) or matrix form.
        """
        cds = self.xp.asarray(cds)
        # all pariwise Atom - atom distances, returned in matrix form
        atm_atm_vec = self.atm_atm_dists(cds)
        # Put on extra dressing for coulomb
        prepped_vec = self.get_prepped_vec(atm_atm_vec)
        # If unsorted and just upper triangle, return prepped_vec
        if not self.sort_mat and not self.full_mat:
            if self.method == 'spf':
                return 1 - self.r_eq / prepped_vec
            else:
                return prepped_vec
        # Otherwise, we need to get full matrix for sorting or to return full matrix
        else:
            # prepped_vec is now the full distance matrix
            prepped_vec = self.dist_matrix(prepped_vec)
            if not self.sort_mat:
                return prepped_vec
            else:
                # First sort by atoms (if necessary), then sort by groups of atoms (if necessary)
                if self.sorted_atoms is not None:
                    prepped_vec = self.sort_atoms(prepped_vec)
                if self.sorted_groups is not None:
                    prepped_vec = self.sort_groups(prepped_vec)
                if self.full_mat:
                    if self.method == 'spf':
                        return 1 - self.r_eq / prepped_vec
                    else:
                        return prepped_vec
                else:
                    prepped_vec = prepped_vec[:, self.idxs_0, self.idxs_1]
                    if self.method == 'spf':
                        return 1 - self.r_eq[self.idxs_0, self.idxs_1] / prepped_vec
                    return prepped_vec
