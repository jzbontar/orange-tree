import os
import ctypes as ct
import numpy as np

_tree = ct.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tree.so'))

class SIMPLE_TREE_NODE(ct.Structure):
    pass

SIMPLE_TREE_NODE._fields_ = [
        ('type', ct.c_int),
        ('children_size', ct.c_int),
        ('split_attr', ct.c_int),
        ('split', ct.c_float),
        ('children', ct.POINTER(ct.POINTER(SIMPLE_TREE_NODE))),
        ('dist', ct.POINTER(ct.c_float)),
        ('n', ct.c_float),
        ('sum', ct.c_float),
    ]

DiscreteNode = 0
ContinuousNode = 1
PredictorNode = 2
Classification = 0
Regression = 1

np.random.seed(42)

N, M = 60, 6
X = np.random.randint(0, 2, (N, M)).astype(np.float64)
y = np.random.randint(0, 2, N).astype(np.float64)
w = np.ones(N).astype(np.float64)
minInstances = 2
maxMajority = ct.c_double(1.0)
maxDepth = 1024
skipProb = ct.c_double(0.0)
type = Classification
cls_vals = 2
attr_vals = (np.max(X, axis=0) + 1).astype(np.int32)
domain = np.zeros(M).astype(np.int32)

X[np.random.random(X.shape) < 0.1] = np.nan
y[np.random.random(y.shape) < 0.1] = np.nan

# create .tab
f = open('/home/jure/tmp/foo.tab', 'w')
f.write('\t'.join('a{}'.format(i) for i in range(M)) + '\tcls\n')
f.write('d\t' * M + 'd\n')
f.write('\t' * M + 'class\n')
for i in range(N):
    f.write('\t'.join('{}'.format('?' if np.isnan(X[i,j]) else int(X[i,j])) for j in range(M)) + '\t{}\n'.format('?' if np.isnan(y[i]) else int(y[i])))


_tree.build_tree.restype = ct.POINTER(SIMPLE_TREE_NODE)
r = _tree.build_tree(
     X.ctypes.data_as(ct.c_void_p),
     y.ctypes.data_as(ct.c_void_p),
     w.ctypes.data_as(ct.c_void_p),
     N, minInstances, maxMajority, maxDepth, skipProb, type, M, cls_vals,
     attr_vals.ctypes.data_as(ct.c_void_p),
     domain.ctypes.data_as(ct.c_void_p))
print(r)
