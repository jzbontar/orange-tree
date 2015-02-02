import os
import ctypes as ct
import numpy as np

_tree = ct.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tree.so'))

class ARGS(ct.Structure):
    _fields_ = [
        ('minInstances', ct.c_int),
        ('maxDepth', ct.c_int),
        ('maxMajority', ct.c_float),
        ('skipProb', ct.c_float),
        ('type', ct.c_int),
        ('attr_split_so_far', ct.POINTER(ct.c_int))
    ]

np.random.seed(42)
# ys = np.random.randint(0, 2, X.shape[0]).astype(np.float32)
# ws = np.ones(X.shape[0]).astype(np.float32)
# X = np.random.normal(0, 1, (20, 3)).astype(np.float32)
# size = X.shape[0]
# M = X.shape[1]
# cls_vals = 2
# attr = 0
# cls_entropy = ct.c_float(0)
# args = ARGS(minInstances=1)
# best_split = ct.c_float(0)
# 
# _tree.gain_ratio_c.restype = ct.c_float
# r = _tree.gain_ratio_c(
#     X.ctypes.data_as(ct.c_void_p), 
#     ys.ctypes.data_as(ct.c_void_p), 
#     ws.ctypes.data_as(ct.c_void_p), 
#     size, M, cls_vals, attr, cls_entropy, 
#     ct.byref(args), ct.byref(best_split))

# N, M = 20, 3
# X = np.random.randint(0, 2, (N, M)).astype(np.float32)
# ys = np.random.randint(0, 2, N).astype(np.float32)
# ws = np.ones(N).astype(np.float32)
# cls_vals = 2
# attr_vals = (np.max(X, axis=0) + 1).astype(np.int)
# attr = 0
# cls_entropy = ct.c_float(0)
# args = ARGS(minInstances=1)
# 
# _tree.gain_ratio_d.restype = ct.c_float
# r = _tree.gain_ratio_d(
#     X.ctypes.data_as(ct.c_void_p), 
#     ys.ctypes.data_as(ct.c_void_p), 
#     ws.ctypes.data_as(ct.c_void_p), 
#     N, M, cls_vals, 
#     attr_vals.ctypes.data_as(ct.c_void_p),
#     attr, cls_entropy, ct.byref(args))

N, M = 20, 3
X = np.random.normal(0, 1, (N, M)).astype(np.float32)
ys = np.random.normal(0, 1, N).astype(np.float32)
ws = np.ones(N).astype(np.float32)
cls_vals = 2
attr_vals = (np.max(X, axis=0) + 1).astype(np.int)
attr = 0
cls_mse = ct.c_float(1)
args = ARGS(minInstances=1)
best_split = ct.c_float(0)

_tree.mse_c.restype = ct.c_float
r = _tree.mse_c(
    X.ctypes.data_as(ct.c_void_p), 
    ys.ctypes.data_as(ct.c_void_p), 
    ws.ctypes.data_as(ct.c_void_p), 
    N, M, attr, cls_mse, 
    ct.byref(args), ct.byref(best_split))
