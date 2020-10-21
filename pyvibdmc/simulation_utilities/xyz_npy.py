import numpy as np


class xyz_npy:
    """Handler of xyz <-> npy conversion, like a personal version of openBabel."""

    @staticmethod
    def extract_xyz(file_name, num_atoms):
        """
        Extracts the coordinates from an xyz file and returns it as an np array of dimension nxmx3,
        where n = number of geometries, m = number of atoms, and 3 = cartesian coordinates

        :param file_name: Name of the .xyz file, including extension
        :type file_name: str
        :param num_atoms: Number of atoms in molecule.
        :type num_atoms: int
        :return: nxmx3 numpy array of coordinates
        """
        block = num_atoms + 3  # every coordinate block has the header (1) +  comments (1) + num atoms (x) + newline (1)
        chunk = np.arange(block)
        atm_chunk = chunk[2:-1]
        cd_list = []
        with open(file_name, 'r') as f:
            lines = f.readlines()
            line_ct = 0
            chunk_cd = np.zeros((num_atoms, 3))
            for line in lines:
                line_stripped = line.strip()  # get rid of whitespace
                if line_ct in atm_chunk:
                    cd_str = line_stripped.split()[1:]
                    chunk_cd[line_ct - 2] = [float(xyorz) for xyorz in cd_str]
                    line_ct += 1
                elif line_ct > np.amax(atm_chunk):
                    line_ct = 0
                    cd_list.append(chunk_cd)
                    chunk_cd = np.zeros((num_atoms, 3))
                else:
                    line_ct += 1
        return np.array(cd_list)


    @staticmethod
    def write_xyz(coords, fname, atm_strings, cmt=None):
        """
        Writes a numpy array of x,y,z coordinates to a .xyz file

        :param fname: name of xyz file
        :param coords: numpy array, either mx3 or nxmx3, where n = number of geometries and m = number of atoms
        :param atm_strings: list of strings that correspond to the atom type e.g. ["H","H","O"]
        :return: np.ndarray
        """
        if len(coords.shape) == 2:
            array = np.expand_dims(coords, axis=0)
        if cmt is None:
            cmt = np.repeat("", len(coords))
        else:
            if not isinstance(cmt, list):
                cmt = np.repeat(cmt, len(coords))
            elif len(cmt) != len(coords):
                raise ValueError
        fl = open(fname, 'w')
        nAtoms = np.shape(coords)[1]
        for mNum, molecule in enumerate(coords):
            fl.write(f"{nAtoms}\n")
            fl.write(f"{cmt[mNum]}\n")
            for atmN, atm in enumerate(molecule):
                fl.write(f"{atm_strings[atmN]} {atm[0]:.14f} {atm[1]:.14f} {atm[2]:.14f}\n")
            fl.write("\n")
        fl.close()
