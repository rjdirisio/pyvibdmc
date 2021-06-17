from collections import defaultdict
import tensorflow as tf
import numpy as np

class TF_Coulomb:
    """Tensorflow implementation of the coulomb matrix by Fenris Lu. Experimental and not documented"""
    def __init__(self, zs, labels=None):
        self.zs = tf.constant(zs, dtype=tf.float64)
        self.n = self.zs.shape[0]
        self.diagonal = 2 * self.zs ** -0.4
        i = np.arange(self.n)
        j = np.arange(self.n)
        I, J = tf.meshgrid(i, j, indexing='ij')
        self.I = tf.reshape(I, (self.n * self.n))
        self.J = tf.reshape(J, (self.n * self.n))
        self.z_map = tf.tensordot(self.zs, self.zs, axes=0)
        atm_dict = defaultdict(list)
        if labels:
            for i in range(len(labels)):
                atm_dict[labels[i]].append(i)
        else:
            for i in range(len(zs)):
                atm_dict[zs[i]].append(i)
        self.atms = atm_dict.values()
        vec_map = 1 - tf.linalg.band_part(tf.constant(np.ones((self.n,self.n))), -1, 0)
        vec_indices = tf.where(vec_map)
        self.vec_indices_flat = vec_indices[:, 0] * self.n + vec_indices[:, 1]

    def coulomb_it(self, cds, sort=True):
        atom_I = tf.gather(cds, self.I, axis=1)
        atom_J = tf.gather(cds, self.J, axis=1)
        d_map = tf.reshape(tf.norm(atom_I-atom_J, axis=2),(cds.shape[0],self.n,self.n))
        d_map += tf.reshape(tf.linalg.tensor_diag(self.diagonal), (1,self.n,self.n))
        coulomb = self.z_map / d_map
        if sort:
            norm = tf.norm(coulomb, axis=1)
            sort_indices = []
            for atm in self.atms:
                if len(atm) == 1:
                    sort_indices.append(tf.constant([atm]*cds.shape[0], dtype=tf.int32))
                else:
                    atm = tf.constant([atm]*cds.shape[0], dtype=tf.int32)
                    with tf.device('/cpu:0'):
                        sub_indices = tf.math.top_k(tf.gather(norm, atm[0], axis=1), k=atm.shape[1]).indices
                    indices = tf.gather(atm, sub_indices, batch_dims=1, axis=1)
                    sort_indices.append(indices)
            sort_indices = tf.concat(sort_indices, axis=1)
            s_coulomb = tf.gather(coulomb, sort_indices, batch_dims=1, axis=1)
            s_coulomb = tf.gather(s_coulomb, sort_indices, batch_dims=1, axis=2)
        else:
            s_coulomb = coulomb
        return s_coulomb

    def vectorize_it(self,coulomb):
        coulomb_flat = tf.reshape(coulomb,(coulomb.shape[0],self.n*self.n))
        return tf.gather(coulomb_flat, self.vec_indices_flat, axis=1)

    def get_coulomb(self, cds, sort=True):
        return self.vectorize_it(self.coulomb_it(cds, sort))