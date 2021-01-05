
import os
from ctypes import *
from os import path
#
# Setup for different clusters
#

cuda_lib = '/home/josef.gugglberger/cuda/lib64/libcudnn.so.7'

if path.exists(cuda_lib):
    lib1 = cdll.LoadLibrary(cuda_lib)
    print('cuda lib loaded ...')
else:
    print(cuda_lib, 'does not exist!')
