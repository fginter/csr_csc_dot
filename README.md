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

Disclaimer: This is an order of magnitude faster than what we could
get from the csr.dot() but maybe some numpy magic would be as fast and
we simply don't know about it.



