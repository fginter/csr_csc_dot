# csr_csc_dot

A fast dot product for sparse matrices where the left matrix is CSR
and the right matrix is CSC, and the result goes into a dense
matrix. So a dot product of (X,M)x(M,Y) where M is much larger than X
and Y. The code supports slices on rows of the first matrix, so you
don't have to produce the whole (X,Y) matrix at once.

Our use case (ie why it was written)

Matrix1 is sparse 1M x 1M matrix.
Matrix2 is sparse 1M x 1M matrix.

And we need a dot product of M1 with M2, which is calculated 500 rows from M1 at a time, against
the whole M2, so 500x1M.

How to compile:

    python3 setup.py build_ext --inplace

How to use:
`csr_csc_dot_f(start,slice,m1,m2,out)` fills `out` with the equivalent of `m1[start:start+slice].dot(m2.T)`
- `m1` is float32 CSR of dimensionality (X,M)
- `m2` is float32 CSC of dimensionality (Y,M)
- `out` is dense float32 of dimensionality (slice,Y)
- `start` is the first row of the slice
- `slice` is slice size, ie how many rows of `m1` will be multiplied
    
Full running example:

    import csr_csc_dot as ccd
    #Get some test matrices
    m1=np.random.rand(5,500)
    m2=np.random.rand(10,500)
    m1_csr=csr_matrix(m1,dtype=np.float32)  #Sparse matrix 1
    m2_csc=csc_matrix(m2,dtype=np.float32)  #Sparse_matrix 2
    out=np.zeros((m1.shape[0],m2.shape[0]),np.float32)
    # fast m1.dot(m2.T)
    ccd.csr_csc_dot_f(0,m1_csr.shape[0],m1_csr,m2_csc,out)
    print("Sparse dot result")
    print(out)




