# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "harm.h":
    void harmOsc (double * in_array, double * out_array, int size)

# create the wrapper code, with numpy type annotations
def getPot(np.ndarray[double, ndim=1, mode="c"] in_array not None,
           np.ndarray[double, ndim=1, mode="c"] out_array not None):

    harmOsc(<double*> np.PyArray_DATA(in_array),
                <double*> np.PyArray_DATA(out_array),
                in_array.shape[0])