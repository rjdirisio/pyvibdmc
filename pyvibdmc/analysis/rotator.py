import numpy as np
import numpy.linalg as la
from .analyze_wfn import AnalyzeWfn

__all__ = ['MolRotator']


class MolRotator:
    """A helper class that will rotate a stack of molecules and generate 3D rotation matrices using vectorized
    numpy operations."""

    @staticmethod
    def rotate_geoms(rot_mats, geoms):
        """Takes in a stack of rotation matrices and applies it to a stack of geometries."""
        new_geoms = np.expand_dims(geoms, -1)  # nxmx3x1
        new_rot_mats = np.expand_dims(rot_mats, 1)  # nx1x3x3
        rot_geoms = np.matmul(new_rot_mats, new_geoms).squeeze()
        if len(rot_geoms.shape) == 2:
            rot_geoms = np.expand_dims(rot_geoms, 0)
        return rot_geoms

    @staticmethod
    def rotate_vec(rot_mats, vecc):
        """Takes in a stack of rotation matrices and applies it to a stack of vector"""
        new_vecc = np.expand_dims(vecc, -1)  # nx3x1
        rot_vecs = np.matmul(rot_mats, new_vecc).squeeze()
        return rot_vecs

    @staticmethod
    def gen_rot_mats(theta, xyz_int):
        """Generates the 3D rotation matrix about X, Y, or Z by theta radians"""
        theta = [theta] if isinstance(theta, float) else theta
        rot_mats = np.zeros((len(theta), 3, 3))
        zeroz = np.zeros(len(theta))
        if xyz_int == 0:  # R_x
            rot_mats[:, 0] = np.tile([1, 0, 0], (len(theta), 1))
            rot_mats[:, 1] = np.column_stack((zeroz, np.cos(theta), -1 * np.sin(theta)))
            rot_mats[:, 2] = np.column_stack((zeroz, np.sin(theta), np.cos(theta)))
        elif xyz_int == 1:  # R_y
            rot_mats[:, 0] = np.column_stack((np.cos(theta), zeroz, -1 * np.sin(theta)))
            rot_mats[:, 1] = np.tile([0, 1, 0], (len(theta), 1))
            rot_mats[:, 2] = np.column_stack((np.sin(theta), zeroz, np.cos(theta)))
        elif xyz_int == 2:  # R_z
            rot_mats[:, 0] = np.column_stack((np.cos(theta), -1 * np.sin(theta), zeroz))
            rot_mats[:, 1] = np.column_stack((np.sin(theta), np.cos(theta), zeroz))
            rot_mats[:, 2] = np.tile([0, 0, 1], (len(theta), 1))
        return rot_mats

    @staticmethod
    def rotate_to_xy_plane(geoms, orig, xax, xyp, return_mat=False):
        """
        Rotate geometries to XY plane, placing one atom at the origin, one on the xaxis,
        and one on the xyplane.  Done through successive rotations about the X-axis, Z-axis then X-axis again.
        """
        if len(geoms.shape) == 2:
            geoms = np.expand_dims(geoms, 0)

        # translation of orig to origin
        geoms = geoms - geoms[:, orig][:, np.newaxis, :]

        # Rotation of xax to x axis
        xax_vec = geoms[:, xax, :]
        x, y, z = [xax_vec[:, num] for num in range(3)]
        theta = np.arctan2(-z, y)
        alpha = np.arctan2((-1 * (y * np.cos(theta) - np.sin(theta) * z)), x)
        r1 = MolRotator.gen_rot_mats(theta, 0)
        r2 = MolRotator.gen_rot_mats(alpha, 2)
        rot_mat = np.matmul(r2, r1)
        geoms = MolRotator.rotate_geoms(rot_mat, geoms)

        # Rotation or xyp to xyplane
        xyz_vec = geoms[:, xyp]
        x, y, z = [xyz_vec[:, num] for num in range(3)]
        beta = np.arctan2(-1 * z, y)
        r3 = MolRotator.gen_rot_mats(beta, 0)
        geoms = MolRotator.rotate_geoms(r3, geoms)
        if return_mat:
            return geoms, r3.dot(r2.dot(r1))
        else:
            return geoms

    @staticmethod
    def gen_eulers(xyz, XYZ):
        """Takes in cartesian vectors and gives you the 3 euler angles that bring xyz to XYZ based on a 'ZYZ'
            rotation"""
        if len(xyz.shape) == 1:
            np.expand_dims(xyz, 0)
        if len(XYZ.shape) == 1:
            np.expand_dims(XYZ, 0)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        X = XYZ[:, 0]
        Y = XYZ[:, 1]
        Z = XYZ[:, 2]

        z_dot = AnalyzeWfn.dot_pdt(z, Z) / (la.norm(z, axis=1) * la.norm(Z, axis=1))
        Yz_dot = AnalyzeWfn.dot_pdt(Y, z) / (la.norm(Y, axis=1) * la.norm(z, axis=1))
        Xz_dot = AnalyzeWfn.dot_pdt(X, z) / (la.norm(X, axis=1) * la.norm(z, axis=1))
        yZ_dot = AnalyzeWfn.dot_pdt(y, Z) / (la.norm(y, axis=1) * la.norm(Z, axis=1))
        xZ_dot = AnalyzeWfn.dot_pdt(x, Z) / (la.norm(x, axis=1) * la.norm(Z, axis=1))
        theta = np.arccos(z_dot)
        phi = np.arctan2(Yz_dot, Xz_dot)
        chi = np.arctan2(yZ_dot, -xZ_dot)  # negative baked in
        return theta, phi, chi

    @staticmethod
    def extract_eulers(rot_mats):
        """From a rotation matrix, calculate the three euler angles theta,phi and Chi. This is based on
            a 'ZYZ' euler rotation"""
        if (rot_mats.shape) == 2:
            rot_mats = np.expand_dims(rot_mats,0)
        zdot = rot_mats[:, -1, -1]
        Yz_dot = rot_mats[:, 2, 1]
        Xz_dot = rot_mats[:, 2, 0]
        yZ_dot = rot_mats[:, 1, 2]
        xZ_dot = rot_mats[:, 0, 2]
        theta = np.arccos(zdot)
        phi = np.arctan2(Yz_dot, Xz_dot)
        chi = np.arctan2(yZ_dot, -xZ_dot)
        return theta, phi, chi
