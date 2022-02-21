from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#integral_utils
#mp2
#help_functions



setup(
    ext_modules=cythonize(["gen_ao_ints.pyx", "scf_helper.pyx" ] , language_level = '3'),
    include_dirs=[numpy.get_include()]
)   