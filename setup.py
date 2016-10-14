from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

e=Extension("csr_csc_dot",["csr_csc_dot.pyx","csr_csc_multiply.c"],include_dirs=[np.get_include()])

setup(
    ext_modules = cythonize([e])
)
