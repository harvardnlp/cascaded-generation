from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext = []
if torch.cuda.is_available():
    ext = [
        CUDAExtension('_genbmm', [
            'matmul_cuda.cpp',
            'matmul_cuda_kernel.cu',
        ]),
    ]

setup(
    name='genbmm',
    version="0.1",
    author="Alexander Rush",
    author_email="arush@cornell.edu",
    packages=["genbmm"],
    ext_modules=ext,
    cmdclass={
        'build_ext': BuildExtension
    })
