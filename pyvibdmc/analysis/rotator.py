import numpy as np
import numpy.linalg as la
from .analyze_wfn import AnalyzeWfn


class MolRotator:
    """A helper class that will rotate a stack of molecules and generate 3D rotation matrices using vectorized
    numpy operations."""

    @staticmethod
    def rotateGeoms(rotMs, geoms):
        """Takes in a stack of rotation matrices and applies it to a stack of geometries."""
        new_geoms = np.expand_dims(geoms, -1)  # nxmx3x1
        new_rotms = np.expand_dims(rotMs, 1)  # nx1x3x3
        rot_geoms = np.matmul(new_rotms, new_geoms).squeeze()
        return rot_geoms

    @staticmethod
    def rotateVector(rotMs, vecc):
        """Takes in a stack of rotation matrices and applies it to a stack of vector"""
        new_vecc = np.expand_dims(vecc, -1)  # nx3x1
        rot_vecs = np.matmul(rotMs, new_vecc).squeeze()
        return rot_vecs

    @staticmethod
    def genXYZ(theta, XYZ):
        """Generates the 3D rotation matrix about X, Y, or Z by theta radians"""
        theta = [theta] if isinstance(theta, float) else theta
        rotM = np.zeros(len(theta), 3, 3)
        zeroLth = np.zeros(len(theta))
        if XYZ == 0:
            rotM[:, 0] = np.tile([1, 0, 0], (len(theta), 1))
            rotM[:, 1] = np.column_stack((zeroLth, np.cos(theta), -1 * np.sin(theta)))
            rotM[:, 2] = np.column_stack((zeroLth, np.sin(theta), np.cos(theta)))
        elif XYZ == 1:
            rotM[:, 0] = np.column_stack((np.cos(theta), zeroLth, -1 * np.sin(theta)))
            rotM[:, 1] = np.tile([0, 1, 0], len(theta), 1)
            rotM[:, 2] = np.column_stack((np.sin(theta), zeroLth, np.cos(theta)))
        elif XYZ == 2:
            rotM[:, 0, :] = np.column_stack((np.cos(theta), -1 * np.sin(theta), zeroLth))
            rotM[:, 1, :] = np.column_stack((np.sin(theta), np.cos(theta), zeroLth))
            rotM[:, 2, :] = np.tile([0, 0, 1], len(theta), 1)
        return rotM

    @staticmethod
    def rotToXYPlane(geoms, orig, xax, xyp, retMat=False):
        """
        Rotate geometries to XY plane, placing one atom at the origin, one on the xaxis,
        and one on the xyplane.  Done through successive rotations about the X-axis, Z-axis then X-axis again.
        """
        if len(geoms.shape) == 2:
            geoms = np.expand_dims(geoms, 0)
        # translation of orig to origin
        geoms -= geoms[:, orig]
        # Rotation of xax to x axis
        xaxVec = geoms[:, xax, :]
        x = xaxVec[:, 0]
        y = xaxVec[:, 1]
        z = xaxVec[:, 2]
        theta = np.arctan2(-z, y)
        alpha = np.arctan2((-1 * (y * np.cos(theta) - np.sin(theta) * z)), x)
        r1 = MolRotator.genXYZ(theta, 0)
        r2 = MolRotator.genXYZ(alpha, 2)
        rotM = np.matmul(r2, r1)
        geoms = MolRotator.rotateGeoms(rotM, geoms)
        # Rotation or xyp to xyplane
        xypVec = geoms[:, xyp]
        z = xypVec[:, 2]
        y = xypVec[:, 1]
        beta = np.arctan2(-1 * z, y)
        r3 = MolRotator.genXYZ(beta, 0)
        geoms = MolRotator.rotateGeoms(r3, geoms)
        if retMat:
            return geoms, r3.dot(r2.dot(r1))
        else:
            return geoms

    @staticmethod
    def genEulers(x, y, z, X, Y, Z):
        """Takes in cartesian vectors and gives you the 3 euler angles that bring xyz to XYZ based on a 'ZYZ'
            rotation"""
        zdot = AnalyzeWfn.dot_pdt(z, Z) / (la.norm(z, axis=1) * la.norm(Z, axis=1))
        Yzdot = AnalyzeWfn.dot_pdt(Y, z) / (la.norm(Y, axis=1) * la.norm(z, axis=1))
        Xzdot = AnalyzeWfn.dot_pdt(X, z) / (la.norm(X, axis=1) * la.norm(z, axis=1))
        yZdot = AnalyzeWfn.dot_pdt(y, Z) / (la.norm(y, axis=1) * la.norm(Z, axis=1))
        xZdot = AnalyzeWfn.dot_pdt(x, Z) / (la.norm(x, axis=1) * la.norm(Z, axis=1))
        Theta = np.arccos(zdot)
        tanPhi = np.arctan2(Yzdot, Xzdot)
        tanChi = np.arctan2(yZdot, -xZdot)  # negative baked in
        return Theta, tanPhi, tanChi

    @staticmethod
    def extractEulers(rotMs):
        """From a rotation matrix, calculate the three euler angles theta,phi and Chi. This is based on
            a 'ZYZ' euler rotation"""
        zdot = rotMs[:, -1, -1]
        Yzdot = rotMs[:, 2, 1]
        Xzdot = rotMs[:, 2, 0]
        yZdot = rotMs[:, 1, 2]
        xZdot = rotMs[:, 0, 2]
        Theta = np.arccos(zdot)
        tanPhi = np.arctan2(Yzdot, Xzdot)
        tanChi = np.arctan2(yZdot, xZdot)
        return Theta, tanPhi, tanChi
