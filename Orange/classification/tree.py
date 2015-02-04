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
IntVar = 0
FloatVar = 1

np.random.seed(42)

N, Mi, Mf = 50, 3, 3
Xi = np.random.randint(0, 2, (N, Mi)).astype(np.float64)
Xf = np.random.normal(0, 2, (N, Mf)).astype(np.float64)
X = np.hstack((Xi, Xf))
y = np.random.normal(0, 2, N).astype(np.float64)
w = np.ones(N).astype(np.float64)
minInstances = 2
maxMajority = ct.c_double(1.0)
maxDepth = 1024
skipProb = ct.c_double(0.0)
type = Regression
cls_vals = 2
attr_vals_i = (np.max(Xi, axis=0) + 1)
attr_vals_f = (np.zeros(Mf))
attr_vals = np.concatenate((attr_vals_i, attr_vals_f)).astype(np.int32)
domain = np.concatenate((np.zeros(Mi), np.ones(Mf))).astype(np.int32)

X[np.random.random(X.shape) < 0.1] = np.nan
y[np.random.random(y.shape) < 0.1] = np.nan

# create .tab
f = open('/home/jure/tmp/foo.tab', 'w')
f.write('\t'.join('a{}'.format(i) for i in range(Mi + Mf)) + '\tcls\n')
f.write('d\t' * Mi + 'c\t' * Mf + '{}\n'.format('d' if type == Classification else 'c'))
f.write('\t' * (Mi + Mf) + 'class\n')
for i in range(N):
    f.write('\t'.join('{}'.format('?' if np.isnan(X[i,j]) else X[i,j]) for j in range(Mi + Mf)) + '\t{}\n'.format('?' if np.isnan(y[i]) else y[i]))

_tree.build_tree.restype = ct.POINTER(SIMPLE_TREE_NODE)
r = _tree.build_tree(
     X.ctypes.data_as(ct.c_void_p),
     y.ctypes.data_as(ct.c_void_p),
     w.ctypes.data_as(ct.c_void_p),
     N, minInstances, maxMajority, maxDepth, skipProb, type, Mi + Mf, cls_vals,
     attr_vals.ctypes.data_as(ct.c_void_p),
     domain.ctypes.data_as(ct.c_void_p))
