import numpy as np
import numpy.linalg as la
import itertools as itt

from .. import Constants
from .. import potential_manager as pm
from .finite_difference import MolFiniteDifference as MolFD


class harmonic_analysis:
    def __init__(self,
                 eq_geom,
                 atoms,
                 potential,
                 masses=None,
                 dx=1.0e-3,
                 points_diag=5,
                 points_off_diag=3):
        """
        A Code that generates the mass weighted hessian matrix (Cartesian coords) and diagonalizes it.
        Uses finite difference to calculate the matrix elements of the hessian.
        :param eq_geom: Numpy array of shape (M, 3), where M is number of atoms long.
        :param atoms: A list of length M specifying the atoms. Can pass in D if deuterium is desired
        :param potential: A potential_manager object from pyvibdmc
        :param potential: Custom masses if needed.
        :param dx: The step size. Defaults to 1e-3 bohr
        :param points_diag: This int specifies the n-point finite difference for the diagonal elements of the hessian.
        Should be [3 or 5]
        :param points_off_diag: This int specifies the n-point finite difference for off diagonals. Should be
        [3 or 5])
        """
        self.eq_geom = eq_geom
        self.atoms = atoms
        self.potential = potential
        self.masses = masses
        self.dx = dx
        self.points_diag = points_diag
        self.points_off_diag = points_off_diag
        self.num_elems = 3 * len(self.atoms)
        self._initialize()

    def _initialize(self):
        if self.masses is None:
            self.masses = np.array([Constants.mass(a) for a in self.atoms])
        # custom masses
        elif isinstance(self.masses, list):
            self.masses = np.array(self.masses)

    def generate_hessian(self):
        num_atoms = len(self.atoms)
        hess = np.zeros((self.num_elems, self.num_elems))

        # Off Diagonals
        # indexing nonsense
        d = np.ravel_multi_index(np.array((np.triu_indices(num_atoms * 3, 1))),
                                 [num_atoms * 3, num_atoms * 3])
        atom_info = np.array((np.unravel_index(d, (num_atoms, 3, num_atoms,
                                                   3)))).T

        # atom_info has information (cd1,atm1,cd2,atm2) which tells the stencil function which atom and dimension to
        # displace
        off_diags = list(itt.combinations(np.arange(self.num_elems), 2))  # indices of upper triangle of hessian
        for k in range(len(off_diags)):
            stencil_cds = MolFD.displace_molecule(eq_geom=self.eq_geom,
                                                  atm_cd=atom_info[k],
                                                  dx=self.dx,
                                                  num_disps=self.points_off_diag,
                                                  )
            potz = self.potential.getpot(stencil_cds)
            potz = np.reshape(potz, (self.points_off_diag,self.points_off_diag))
            hess[off_diags[k]] = MolFD.differentiate(values=potz,
                                                     dx=self.dx,
                                                     num_points=self.points_off_diag,
                                                     der=11)
        hess = hess + hess.T  # hessian is symmetric matrix

        # On Diagonals
        s = [np.arange(num_atoms), np.arange(3)]
        atom_info = list(itt.product(*s))
        for onDiags in range(self.num_elems):
            stencil_cds = MolFD.displace_molecule(eq_geom=self.eq_geom,
                                                  atm_cd=atom_info[onDiags],
                                                  dx=self.dx,
                                                  num_disps=self.points_diag
                                                  )
            hess[onDiags, onDiags] = MolFD.differentiate(values=self.potential.getpot(stencil_cds),
                                                         dx=self.dx,
                                                         num_points=self.points_diag,
                                                         der=2
                                                         )
        return hess

    def diagonalize(self, hessian):
        masses_dup = np.repeat(self.masses, 3)
        hess_mw = hessian / np.sqrt(masses_dup) / np.sqrt(masses_dup[:, np.newaxis])
        freqs, normal_modes = la.eigh(hess_mw)
        freqs_cm = Constants.convert(np.sqrt(np.abs(freqs)) * np.sign(freqs), 'wavenumbers', to_AU=False)
        return freqs_cm, normal_modes

    def run(self):
        hessian = self.generate_hessian()
        freqs_cm, normal_modes = self.diagonalize(hessian)
        return freqs_cm, normal_modes
