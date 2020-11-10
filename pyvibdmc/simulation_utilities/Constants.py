massDict = {'H': 1.00782503, 'D': 2.01410178, 'T': 3.01604928, 'He': 4.00260325, 'Li': 7.01600344, 'Be': 9.01218306,
            'B': 11.00930536, 'C': 12.0, 'N': 14.003074, 'O': 15.99491462, 'F': 18.99840316, 'Ne': 19.99244018,
            'Na': 22.98976928,
            'Mg': 23.9850417, 'Al': 26.98153853, 'Si': 27.97692653, 'P': 30.973762, 'S': 31.97207117, 'Cl': 34.96885268,
            'Ar': 39.96238312, 'K': 38.96370649, 'Ca': 39.96259086, 'Sc': 44.95590828, 'Ti': 47.94794198,
            'V': 50.94395704,
            'Cr': 51.94050623, 'Mn': 54.93804391, 'Fe': 55.93493633, 'Co': 58.93319429, 'Ni': 57.93534241,
            'Cu': 62.92959772,
            'Zn': 63.92914201, 'Ga': 68.9255735, 'Ge': 73.92117776, 'As': 74.92159457, 'Se': 79.9165218,
            'Br': 78.9183376,
            'Kr': 83.91149773, 'Rb': 84.91178974, 'Sr': 87.9056125, 'Y': 88.9058403, 'Zr': 89.9046977, 'Nb': 92.906373,
            'Mo': 97.90540482, 'Tc': 96.9063667, 'Ru': 101.9043441, 'Rh': 102.905498, 'Pd': 105.9034804,
            'Ag': 106.9050916,
            'Cd': 113.90336509, 'In': 114.90387878, 'Sn': 119.90220163, 'Sb': 120.903812, 'Te': 129.90622275,
            'I': 126.9044719,
            'Xe': 131.90415509, 'Cs': 132.90545196, 'Ba': 137.905247, 'La': 138.9063563, 'Ce': 139.9054431,
            'Pr': 140.9076576,
            'Nd': 141.907729, 'Pm': 144.9127559, 'Sm': 151.9197397, 'Eu': 152.921238, 'Gd': 157.9241123,
            'Tb': 158.9253547,
            'Dy': 163.9291819, 'Ho': 164.9303288, 'Er': 165.9302995, 'Tm': 168.9342179, 'Yb': 173.9388664,
            'Lu': 174.9407752,
            'Hf': 179.946557, 'Ta': 180.9479958, 'W': 183.95093092, 'Re': 186.9557501, 'Os': 191.961477,
            'Ir': 192.9629216,
            'Pt': 194.9647917, 'Au': 196.96656879, 'Hg': 201.9706434, 'Tl': 204.9744278, 'Pb': 207.9766525,
            'Bi': 208.9803991,
            'Po': 208.9824308, 'At': 209.9871479, 'Rn': 210.9906011, 'Fr': 223.019736, 'Ra': 223.0185023,
            'Ac': 227.0277523,
            'Th': 232.0380558, 'Pa': 231.0358842, 'U': 238.0507884, 'Np': 236.04657, 'Pu': 238.0495601,
            'Am': 241.0568293,
            'Cm': 243.0613893, 'Bk': 247.0703073, 'Cf': 249.0748539, 'Es': 252.08298, 'Fm': 257.0951061,
            'Md': 258.0984315,
            'No': 259.10103, 'Lr': 262.10961, 'Rf': 267.12179, 'Db': 268.12567, 'Sg': 271.13393, 'Bh': 272.13826,
            'Hs': 270.13429,
            'Mt': 276.15159, 'Ds': 281.16451, 'Rg': 280.16514, 'Cn': 285.17712, 'Nh': 284.17873, 'Fl': 289.19042,
            'Mc': 288.19274,
            'Lv': 293.20449, 'Ts': 292.20746, 'Og': 294.21392}


def get_atomic_num(atms):
    """
    :param atms: A list (or single string) of atomic element symbols
    :return: The atomic numbers of each of the atom strings you provide
    """
    atm_strings = list(massDict.keys())
    return [atm_strings.index(n) + 1 for n in atms]


def get_atomic_string(atomic_num):
    """
    :param atomic_num: The atomic numbers of each of the atom strings you provide
    :return: A list of atomic element symbols
    """
    if type(atomic_num) is not list: atomic_num = [atomic_num]
    atm_strings = list(massDict.keys())
    return [atm_strings[anum] for anum in atomic_num]


class Constants:
    """
    Thanks, Mark Boyer, for this silly little class.
    Converter that handles energy, distance, and mass conversions for DMC. Can be expanded upon.
    """
    atomic_units = {
        "wavenumbers": 4.556335281212229e-6,
        "angstroms": 1 / 0.529177,
        "amu": 1.000000000000000000 / 6.02213670000e23 / 9.10938970000e-28  # 1822.88839  g/mol -> a.u.
    }

    @classmethod
    def convert(cls, val, unit, to_AU=True):
        """
        :param val: The value or values that will be converted
        :type val: np.ndarray
        :param unit: The units (not atomic units) that we will be converting to or from
        :type unit: str
        :param to_AU: If true, converting from non-a.u. to a.u.  If false, converting to a.u. from non-a.u.
        :type to_AU:boolean
        :return: converted values
        """
        vv = cls.atomic_units[unit]
        return (val * vv) if to_AU else (val / vv)

    @classmethod
    def mass(cls, atom, to_AU=True):
        """
        Given a string that corresponds to an atomic element, output the atomic mass of that element
        :param atom: The string of an atomic element
        :type atom:str
        :param to_AU: If true, converting from non-a.u. to a.u.  If false, converting to a.u. from non-a.u.
        :type to_AU:boolean
        :return: mass in atomic units unless user changes to_AU to False, then AMU
        """
        m = massDict[atom]
        if to_AU:
            m = cls.convert(m, 'amu')
        return m

    @classmethod
    def reduced_mass(cls, atoms, to_AU=True):
        """
                Given a string like 'O-H' or 'N-N' , output the reduced mass of that diatomic
                :param atoms: A string that is composed of two atoms
                :type atom:str
                :param to_AU: If true, converting from non-a.u. to a.u.  If false, converting to a.u. from non-a.u.
                :type to_AU:boolean
                :return: mass in atomic units unless user changes to_AU to False, then AMU
                """
        atoms = atoms.split('-')
        atm1 = atoms[0]
        atm2 = atoms[1]
        mass1 = massDict[atm1]
        mass2 = massDict[atm2]
        if to_AU:
            mass1 = cls.convert(mass1, 'amu')
            mass2 = cls.convert(mass2, 'amu')
        reduced_mass = mass1 * mass2 / (mass1 + mass2)
        return reduced_mass