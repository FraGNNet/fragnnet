import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        [
            "src/fragnnet/frag/compute_frags.pyx",
            "src/fragnnet/frag/multi_cut_bfs.pyx",
            "src/fragnnet/massformer/algos.pyx",
        ],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
)
