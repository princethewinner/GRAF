#!/usr/bin/env python3
#encoding: UTF-8

from distutils.core import setup, Extension
from Cython.Build import cythonize

import os

print('Modified code')

os.environ['CFLAGS'] = ' -g -std=c++0x -D__DEBUG_TREE=false -D__DEBUG_HASH=false'
setup(ext_modules = cythonize(Extension('GRAFPlaneSearchGPUROI', sources=['GRAF.pyx', 'GuidedRandomForest.cpp'], language='c++', library_dirs=['/usr/local/cuda-10.0/targets/x86_64-linux/lib/'], libraries=['cudart'], extra_objects=['CudaUtil.o'])), version=1.0, authon='Aashi Jindal & Prashant Gupta', name='GRAFPlaneSearchGPUROI')
