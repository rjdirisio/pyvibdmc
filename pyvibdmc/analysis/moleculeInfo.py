import numpy as np
import numpy.linalg as la

class molecInfo:
    def __init__(self,coordinates):
        """Takes xyz coordinates of an molecule, and calculates attributes associated with the molecule. This is built
        with DMC wave functions in mind, but can be used for any xyz atom. This class handles bond lengths, angles, etc.

        :param coordinates: mx3 array, or nxmx3 array, n = number of geometries, m = number of atoms, 3 is x y z
        :type coordinates: np.ndarray"""
        self.xx = coordinates
        if len(self.xx.shape) == 2: #if only one geometry, make two
            self.xx = np.expand_dims(self.xx,axis=0)

    @staticmethod
    def dotPdt(v1,v2):
        """Takes two stacks of vectors and dots the pairs of vectors together"""
        new_v1 = np.expand_dims(v1,axis=1)
        new_v2 = np.expand_dims(v2,axis=2)
        return np.matmul(new_v1, new_v2).squeeze()

    @staticmethod
    def expVal(operator,dw):
        """DMC equivalent of <\psi_0 | A | \psi_0>"""
        if len(operator.shape) > 1:
            return np.average(operator,axis=0,weights=dw)
        else:
            return np.average(operator,weights=dw)


    def bondLength(self, atm1, atm2):
        """
        Computes the norm of the vector between two atoms.
        :param atm1: desired atom number 1  (python index starts at 0)
        :type atm1: int
        :param atm2: desired atom number 2
        :type atm2: int
        :return: norm of x[atm1]-x[atm2]
        """

        return la.norm(self.xx[:,atm1] - self.xx[:,atm2],axis=1)

    def bondAngle(self, atm1,atmMid,atm3):
        """Calculate atm1 - atmMid - atm3 angle"""
        vec1 = self.xx[:,atm1] - self.xx[:,atmMid]
        vec2 = self.xx[:,atm3] - self.xx[:,atmMid]
        #cos(x) = left.right/(|left|*|right|)
        dotV = np.arccos(self.dotPdt(vec1,vec2) /
                         (la.norm(vec1,axis=1)*la.norm(vec2,axis=1)))
        return dotV

    def bisectingVector(self,atm1,atmM,atm3):
        """Get  normalized bisecting vector of two vectors (two bond lengths).
        It is assumed that the two flanking atoms have the same central atom.
        |b|*a + |a|*b"""
        b1 = self.xx[:,atm1] - self.xx[:,atmM]
        b2 = self.xx[:,atm3] - self.xx[:,atmM]
        bis = la.norm(b1,axis=1) * b2 + \
              la.norm(b2,axis=1) * b1
        return bis / la.norm(bis,axis=1)

    def dihedral(self,vec1,vec2,vec3):
        """Looking down vec2, calculate dihedral angle
        # https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
        """
        crossterm1 = np.cross(np.cross(vec1, vec2, axis=1), np.cross(vec2, vec3, axis=1), axis=1)
        term1 = self.dotPdt(crossterm1 , (vec2 / la.norm(vec2, axis=1)[:, np.newaxis]))
        term2 = self.dotPdt(np.cross(vec1, vec2, axis=1) , np.cross(vec2, vec3, axis=1))
        dh = np.arctan2(term1, term2)
        return dh

    def getComponent(self,atm,xyz):
        """Get x, y, or z component of a vector that corresponds to a particular atom in the predetermined cooridnate
        system.
        :param atm: atom's index
        :type: int
        :param xyz: Either 0 (x), 1 (y), or 2 (z)
        :type xyz: int
        :return: vector of x, y, or z components of a stack of vectors
        """
        return self.xx[:,atm,xyz]

    @staticmethod
    def cartToSpherical(vecc):
        """Takes stack of vectors and translates them to stack of r theta phi coordinates,
        with the assumption that you are already in the coordinate system you desire."""
        r = la.norm(vecc, axis=1)
        th = np.arccos(vecc[:, -1] / r)  #z/r
        phi = np.arctan2(vecc[:, 1], vecc[:, 0]) #y/x
        return r,th,phi
        
    @staticmethod    
    def projection_1D(attr,descWeights,binNum=25,range=None,normalize=True):
        """
        Project the probability amplitude onto a particular coordinate; 1D Histogram.
        :param attr: the coordinate to be projected onto
        :param descWeights: the descendant weights for \psi**2
        :param binNum: Number of bins in histogram (higher number, more number of walkers needed)
        :param range: Range over which attr is histogrammed
        :param normalize: Normalizes the wave function along coordinate
        :return: x,y (bin centers, amplitude at bins)
        """
        amp,binEdge = np.histogram(attr,bins=binNum, range=range, weights=descWeights, density=normalize)
        xx = 0.5*(binEdge[1:]+binEdge[:-1]) #get bin centers
        return np.column_stack((xx,amp))

    @staticmethod    
    def projection_2D():
        raise NotImplementedError
    
    
    
    