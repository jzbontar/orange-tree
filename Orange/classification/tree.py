import os
import ctypes as ct
import numpy as np

from Orange.classification.base import Learner, Model

_tree = ct.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tree.so'))

DiscreteNode = 0
ContinuousNode = 1
PredictorNode = 2
Classification = 0
Regression = 1
IntVar = 0
FloatVar = 1

c_int_p = ct.POINTER(ct.c_int)
c_double_p = ct.POINTER(ct.c_double)

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

_tree.build_tree.restype = ct.POINTER(SIMPLE_TREE_NODE)
_tree.new_node.restype = ct.POINTER(SIMPLE_TREE_NODE)

class SimpleTreeNode:
    pass

class SimpleTreeLearner(Learner):
    def __init__(self, minInstances=2, maxDepth=1024, maxMajority=1.0, skipProb=0.0):
        self.minInstances = minInstances
        self.maxDepth = maxDepth
        self.maxMajority = maxMajority
        self.skipProb = skipProb
    
    def fit_storage(self, data):
        return SimpleTreeModel(self, data)

class SimpleTreeModel(Model):
    def __init__(self, learner, data):
        self.num_attrs = data.X.shape[1]
        if isinstance(data.domain.class_var, Orange.data.DiscreteVariable):
            self.type = Classification
            self.cls_vals = len(data.domain.class_var.values)
        elif isinstance(data.domain.class_var, Orange.data.ContinuousVariable):
            self.type = Regression
            self.cls_vals = 0
        else:
            assert(False)
        attr_vals = []
        domain = []
        for attr in data.domain.attributes:
            if isinstance(attr, Orange.data.DiscreteVariable):
                attr_vals.append(len(attr.values))
                domain.append(IntVar)
            elif isinstance(attr, Orange.data.ContinuousVariable):
                attr_vals.append(0)
                domain.append(FloatVar)
            else:
                assert(False)
        attr_vals = np.array(attr_vals, dtype=np.int32)
        domain = np.array(domain, dtype=np.int32)
        self.node = _tree.build_tree(
            data.X.ctypes.data_as(c_double_p),
            data.Y.ctypes.data_as(c_double_p),
            data.W.ctypes.data_as(c_double_p),
            data.X.shape[0], 
            data.W.size,
            learner.minInstances,
            learner.maxDepth,
            ct.c_float(learner.maxMajority),
            ct.c_float(learner.skipProb),
            self.type,
            self.num_attrs,
            self.cls_vals,
            attr_vals.ctypes.data_as(c_int_p),
            domain.ctypes.data_as(c_int_p))

    def predict(self, X):
        if self.type == Classification:
            p = np.zeros((X.shape[0], self.cls_vals))
            _tree.predict_classification(
                X.ctypes.data_as(c_double_p),
                X.shape[0],
                self.node,
                self.num_attrs,
                self.cls_vals,
                p.ctypes.data_as(c_double_p))
            return p.argmax(axis=1), p
        elif self.type == Regression:
            p = np.zeros(X.shape[0])
            _tree.predict_regression(
                X.ctypes.data_as(c_double_p),
                X.shape[0],
                self.node,
                self.num_attrs,
                p.ctypes.data_as(c_double_p))
            return p
        else:
            assert(False)

    def __del__(self):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self):
        pass

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
        return py_node

    def __from_python(self, py_node):
        node = _tree.new_node(py_node.children_size, self.type, self.cls_vals)
        n = node.contents
        n.type = py_node.type
        n.children_size = py_node.children_size
        n.split_attr = py_node.split_attr
        n.split = py_node.split
        for i in range(n.children_size):
            n.children[i] = self.__from_python(py_node.children[i])
        if self.type == Classification:
            for i in range(self.cls_vals):
                n.dist[i] = py_node.dist[i]
        else:
            n.n = py_node.n
            n.sum = py_node.sum
        return node
        
if __name__ == '__main__':
    import Orange
    np.random.seed(42)
    
    type = Regression
    
    N, Mi, Mf = 50, 6, 2
    Xi = np.random.randint(0, 2, (N, Mi)).astype(np.float64)
    Xf = np.random.normal(0, 2, (N, Mf)).astype(np.float64)
    X = np.hstack((Xi, Xf))
    if type == Classification:
        y = np.random.randint(0, 2, N).astype(np.float64)
    else:
        y = np.random.normal(0, 2, N).astype(np.float64)
    
    # X[np.random.random(X.shape) < 0.1] = np.nan
    # y[np.random.random(y.shape) < 0.1] = np.nan
    
    # create .tab
    f = open('/home/jure/tmp/foo.tab', 'w')
    f.write('\t'.join('a{}'.format(i) for i in range(Mi + Mf)) + '\tcls\n')
    f.write('d\t' * Mi + 'c\t' * Mf + '{}\n'.format('d' if type == Classification else 'c'))
    f.write('\t' * (Mi + Mf) + 'class\n')
    for i in range(N):
        f.write('\t'.join('{}'.format('?' if np.isnan(X[i,j]) else X[i,j]) for j in range(Mi + Mf)) + '\t{}\n'.format('?' if np.isnan(y[i]) else y[i]))
    f.close()
    
    _data = Orange.data.Table('/home/jure/tmp/foo.tab')
    
    learner = SimpleTreeLearner()
    model = learner(_data)
    p = model(_data)
    for pp in p:
        print(pp)

