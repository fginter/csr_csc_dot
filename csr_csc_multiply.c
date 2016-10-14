/*
M1
(3, 3001)
rdata [ 1.  1.  1.  1.  1.]
rindices [1000 2000 3000 1000 3000]
rindptr [0 1 3 5]

M2
(3, 3001)
data [ 1.  1.  1.  1.  1.]
indices [0 2 1 1 2]
indptr [0 0 0 ..., 3 3 5]
*/

#include <stdint.h>

typedef int32_t idx_type;
typedef float val_type;

void fast_dot(idx_type m1_start, idx_type m1_slice, idx_type *m1_row_ptr, idx_type *m1_indices, val_type *m1_data, idx_type *m2_col_ptr, idx_type *m2_indices, val_type *m2_data, val_type *out, idx_type out_col_number) {
    idx_type out_row,col,out_col,m1_ri,m1_ci,m2_ri;
    val_type m1_val,m2_val;
    
    for (m1_ri=m1_start; m1_ri<m1_start+m1_slice; m1_ri++) { //for every row of m1
	out_row=m1_ri-m1_start;
	
	for (m1_ci=m1_row_ptr[m1_ri]; m1_ci<m1_row_ptr[m1_ri+1]; m1_ci++) { //for every column on that row
	    col=m1_indices[m1_ci]; //this is now a column index in m2
	    m1_val=m1_data[m1_ci]; //the corresponding value from m1

	    for (m2_ri=m2_col_ptr[col]; m2_ri<m2_col_ptr[col+1]; m2_ri++) { //for every row in m2 in this column
		out_col=m2_indices[m2_ri];
		m2_val=m2_data[m2_ri];

		out[out_col_number*out_row+out_col]+=m1_val*m2_val;
	    }
	}
    }//m1 row index loop
}
