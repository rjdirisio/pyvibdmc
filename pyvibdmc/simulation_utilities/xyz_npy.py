__all__ = ['xyz_npy']
import numpy as np
class xyz_npy:
    """This short class handles xyz <-> npy conversion, like a manual version of openBabel. """
    @staticmethod
    def extractXYZ(fname,atmStr):
        import pandas as pd
        """
        Extracts the coordinates from an xyz file and returns it as an np array of dimension nxmx3,
        where n = number of geometries, m = number of atoms, and 3 = cartesian coordinates
        :param fname:
        :type str
        :param atmStr: a list of strings that will be used to parse the file.
        :type str        
        :return: np.ndarray
        """
        k =  pd.read_table(fname,delim_whitespace=True,names = ['atom','x','y','z'])
        k.dropna(inplace=True)
        someVals = list(set(atmStr))
        j=k.loc[k['atom'].isin(someVals)] 
        xx=j.loc[:,['x','y','z']].to_numpy()
        nLines=len(xx)
        nAts = len(atmStr)
        return xx.reshape(int(nLines/nAts),nAts,3)

    @staticmethod
    def writeXYZ(xx, fname,atmStrings,cmt=None):
        """
        Writes a numpy array of x,y,z coordinates to a .xyz file
        :param fname:name of xyz file
        :type str
        :param xx:numpy array, either mx3 or nxmx3, where n = number of geometries and m = number of atoms
        :param atmStrings: list of strings that correspond to the atom type e.g. ["H","H","O"]
        :return: np.ndarray        
        """
        if len(xx.shape) == 2:
            array = np.expand_dims(xx,axis=0)
        if cmt is None:
            cmt = np.repeat("",len(xx))
        else:
            if len(cmt) == 1:
                cmt = np.repeat(cmt,len(xx))
            else:
                raise ValueError('Comment length does not match array length')
        fl = open(fname,'w')
        nAtoms = np.shape(xx)[1]
        for mNum, molecule in enumerate(xx):
            fl.write(f"{nAtoms}\n")
            fl.write(f"{cmt[mNum]}\n")
            for atmN,atm in enumerate(molecule):
                fl.write(f"{atmStrings[atmN]} {atm[0]:.14f} {atm[1]:.14f} {atm[2]:.14f}\n")
            fl.write("\n")
        fl.close()
        
# if __name__ == '__main__':
#     fll = 'test.xyz'
#     atmList = ["O","O","O","H","H","H","H","H","H","H"]
#     xx = fileHandler.extractXYZ(fll,atmList)
#     fileHandler.writeXYZ(xx,'myExit.xyz',atmList)