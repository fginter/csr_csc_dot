#cython: language_level=3

import numpy as np
cimport numpy as np
import sys

ctypedef np.int32_t idx_type;
ctypedef np.float32_t val_type;

cdef extern void fast_dot(idx_type m1_start, idx_type m1_slice_len, idx_type *m1_row_ptr, idx_type *m1_indices, idx_type *m2_col_ptr, idx_type *m2_indices, val_type *out, idx_type out_col_number)

#m1 is CSR
#m2 is CSC
#out is dense float32
#start is slice beginning
#slice is slice size

def csr_csc_dot_f(start,slice,m1,m2, np.ndarray[np.float32_t,ndim=2,mode="c"] out):

    if start+slice > out.shape[0]:
       print("csr_csc_dot: adjusting slice to ",out.shape[0]-start,file=sys.stderr)
       slice=out.shape[0]-start

    cdef np.ndarray[np.int32_t,ndim=1,mode="c"] m1_i=m1.indices
    cdef idx_type * m1_idx=<np.int32_t *> m1_i.data

    cdef np.ndarray[np.int32_t,ndim=1,mode="c"] m1_p=m1.indptr
    cdef idx_type * m1_ptr=<np.int32_t *> m1_p.data

    cdef np.ndarray[np.int32_t,ndim=1,mode="c"] m2_i=m2.indices
    cdef idx_type * m2_idx=<np.int32_t *> m2_i.data

    cdef np.ndarray[np.int32_t,ndim=1,mode="c"] m2_p=m2.indptr
    cdef idx_type * m2_ptr=<np.int32_t *> m2_p.data

    cdef val_type * out_data=<np.float32_t *> out.data

    out.fill(0.0)
    fast_dot(start,slice,m1_ptr,m1_idx,m2_ptr,m2_idx,out_data,out.shape[1])

