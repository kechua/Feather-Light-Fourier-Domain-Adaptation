import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

module_path = os.path.dirname(__file__)

setup(
    name='fused',
    ext_modules=[
        CUDAExtension('fused_cuda', [
           os.path.join(module_path, 'fused_bias_act.cpp'),
           os.path.join(module_path, 'fused_bias_act_kernel.cu'),
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='upfirdn2d',
    ext_modules=[
        CUDAExtension('upfirdn2d_cuda', [
           os.path.join(module_path, 'upfirdn2d.cpp'),
           os.path.join(module_path, 'upfirdn2d_kernel.cu'),
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)


