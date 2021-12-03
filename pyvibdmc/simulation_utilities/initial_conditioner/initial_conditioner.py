import numpy as np
from ..Constants import *

__all__ = ['InitialConditioner']


class InitialConditioner:
    """
    If given a minimum energy geometry of the system you are trying to run, it will generate a preliminary ensemble.
    In one instance, you can calculate the harmonic frequencies and normal modes, and then sample the harmonic 3N-6
    ground state wave function along those  normal modes. Will, in the future, handle other initial conditions.
    """

    def __init__(self, coord, atoms, num_walkers, technique, technique_kwargs, masses=None):
        self.coord = coord
        self.atoms = atoms
        self.masses = masses
        self.num_walkers = num_walkers
        self.technique = technique.lower()
        self.technique_kwargs = technique_kwargs
        self._initialize()

    def _initialize(self):
        if self.technique == 'harmonic_sampling':
            self.run_func = self.run_harm
        elif self.technique == 'permute_atoms':
            self.run_func = self.run_permute
        else:
            raise NotImplementedError("Choose another initial condition: harmonic_sampling or permute_atoms")

    def gen_disps(self, sigmas):
        disps = np.random.normal(loc=0,
                                 scale=sigmas,
                                 size=(self.num_walkers, len(sigmas)))
        return disps

    def displace_along_nms(self, freqz, nmz, massez, ensemble):
        three_n_6 = len(freqz)
        three_n = three_n_6 + 6

        # Generate normal mode coordinates based on frequency (atomic_units)
        sigmas = 1 / np.sqrt(freqz)
        displaced_nms = self.gen_disps(sigmas)

        # Dot the displaced normal mode vals into the eigenvector "L" matrix to get back mass weighted cart coordinates.
        nmz_t = nmz.T
        displaced_carts = np.matmul(displaced_nms, nmz_t[np.newaxis, :, :]).squeeze()

        # Reshape the cartesian coordinates to resemble how the walkers are constructed (n, m ,3) m = num_atoms
        displaced_carts = np.reshape(displaced_carts, (-1, three_n // 3, 3))

        # Divide through by the sqrt of the masses to scale from mass weighted back to regular cartesian
        displaced_carts = displaced_carts / np.sqrt(massez)[:, np.newaxis]

        # Displace the walkers along those normal modes
        if ensemble is None:
            displaced_cds = np.tile(self.coord, (self.num_walkers, 1, 1))
        else:
            displaced_cds = ensemble
        displaced_cds = displaced_cds + displaced_carts

        return displaced_cds

    def run_harm(self):
        freqs = self.technique_kwargs['freqs']
        nms = self.technique_kwargs['normal_modes']
        scaling = self.technique_kwargs['scaling_factor']
        ensemble = self.technique_kwargs['ensemble']

        if self.masses is None:
            mass_prelim = np.array([Constants.mass(a) for a in self.atoms])
        else:
            mass_prelim = self.masses

        # Extract the vibrations from the normal mode calculation, getting rid of the 6 smallest eigenvals/vecs
        freqs_3n_6 = freqs[6:]
        nms_3n_6 = nms[:, 6:]

        # Convert freqs to au scale according to input
        freqs_au = Constants.convert(freqs_3n_6, 'wavenumbers', to_AU=True)
        freqs_au = freqs_au / scaling
        displaced_cds = self.displace_along_nms(freqs_au, nms_3n_6, mass_prelim, ensemble)
        return displaced_cds

    def run_permute(self):
        """Must pass in a list of lists."""
        # CH5+ = [[1,2,3,4,5]] , or for testing [[1,2,3],[4,5]]
        # H5O2+ = [[2,3],[4,5]] [O_left,O_left,H_left,H_left,H_right,H_right,H_center]
        like_atoms = self.technique_kwargs['like_atoms']
        ensemble = self.technique_kwargs['ensemble']

        # Get ensemble size
        if ensemble is None:
            walkers = np.tile(self.coord, (self.num_walkers, 1, 1))
        else:
            walkers = ensemble

        # For each tuple of like atoms, we will randomly permute them
        for pair in like_atoms:
            cds_to_randomize = walkers[:, pair]
            [np.random.shuffle(x) for x in cds_to_randomize]
            # Assign the stack of permuted atom coordinates to the appropriate place in the walker array
            walkers[:, pair] = cds_to_randomize
        return walkers

    def run(self):
        new_coords = self.run_func()
        return new_coords
