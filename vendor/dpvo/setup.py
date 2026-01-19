"""
Setup script to build DPVO CUDA extensions.

Run from the isolated environment:
    cd vendor/dpvo && python setup.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory where this setup.py is located
ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name='dpvo_cuda_extensions',
    ext_modules=[
        CUDAExtension(
            'cuda_corr',
            sources=[
                os.path.join(ROOT, 'altcorr', 'correlation.cpp'),
                os.path.join(ROOT, 'altcorr', 'correlation_kernel.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            }
        ),
        CUDAExtension(
            'cuda_ba',
            sources=[
                os.path.join(ROOT, 'fastba', 'ba.cpp'),
                os.path.join(ROOT, 'fastba', 'ba_cuda.cu'),
                os.path.join(ROOT, 'fastba', 'block_e.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
