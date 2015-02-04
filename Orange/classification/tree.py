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
        ('attr_split_so_far', ct.POINTER(ct.c_int)),
        ('num_attrs', ct.c_int),
        ('cls_vals', ct.c_int),
        ('attr_vals', ct.POINTER(ct.c_int)),
        ('domain', ct.POINTER(ct.c_int)),
    ]

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

class SimpleTreeNode:
    pass

class SimpleTree:
    def __init__(self):
        self.cls_vals = 2
        self.type = Regression

    def __to_python(self, node):
        n = node.contents
        py_node = SimpleTreeNode()
        py_node.type = n.type
        py_node.children_size = n.children_size
        py_node.split_attr = n.split_attr
        py_node.split = n.split
        py_node.children = [self.__to_python(n.children[i]) for i in range(n.children_size)]
        if self.type == Classification:
            py_node.dist = [n.dist[i] for i in range(self.cls_vals)]
        else:
            py_node.n = n.n
            py_node.sum = n.sum

    def __from_python(self, py_node):
        pass
        
        

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
y = np.random.randint(0, 2, N).astype(np.float64)
# y = np.random.normal(0, 2, N).astype(np.float64)
w = np.ones(N).astype(np.float64)
attr_vals_i = (np.max(Xi, axis=0) + 1)
attr_vals_f = (np.zeros(Mf))
attr_vals = np.concatenate((attr_vals_i, attr_vals_f)).astype(np.int32)
domain = np.concatenate((np.zeros(Mi), np.ones(Mf))).astype(np.int32)

args = ARGS()
args.minInstances = 2
args.maxMajority = 1.0
args.maxDepth = 1024
args.skipProb = 0.0
args.type = Classification
args.num_attrs = Mi + Mf
args.cls_vals = 2
args.attr_vals = attr_vals.ctypes.data_as(ct.POINTER(ct.c_int))
args.domain = domain.ctypes.data_as(ct.POINTER(ct.c_int))

# X[np.random.random(X.shape) < 0.1] = np.nan
# y[np.random.random(y.shape) < 0.1] = np.nan

# create .tab
f = open('/home/jure/tmp/foo.tab', 'w')
f.write('\t'.join('a{}'.format(i) for i in range(Mi + Mf)) + '\tcls\n')
f.write('d\t' * Mi + 'c\t' * Mf + '{}\n'.format('d' if args.type == Classification else 'c'))
f.write('\t' * (Mi + Mf) + 'class\n')
for i in range(N):
    f.write('\t'.join('{}'.format('?' if np.isnan(X[i,j]) else X[i,j]) for j in range(Mi + Mf)) + '\t{}\n'.format('?' if np.isnan(y[i]) else y[i]))
f.close()

_tree.build_tree.restype = ct.POINTER(SIMPLE_TREE_NODE)
node = _tree.build_tree(
    X.ctypes.data_as(ct.c_void_p),
    y.ctypes.data_as(ct.c_void_p),
    w.ctypes.data_as(ct.c_void_p),
    N, ct.byref(args))

t = SimpleTree()
print(t._SimpleTree__to_python(node))

# p = np.zeros(N)
# _tree.predict_regression(
#     X.ctypes.data_as(ct.c_void_p),
#     N, node, ct.byref(args),
#     p.ctypes.data_as(ct.c_void_p))
# 
