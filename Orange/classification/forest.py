import numpy as np

import Orange

class RandomForestLearner(Orange.classification.base.Learner):
    def __init__(self, n_estimators=10, skip_prob='sqrt', max_depth=1024,
                 min_instances=2, max_majority=1.0):
        self.n_estimators = n_estimators
        self.skip_prob = skip_prob
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.max_majority = max_majority

    def fit_storage(self, data):
        return RandomForestModel(self, data)

class RandomForestModel(Orange.classification.base.Model):
    def __init__(self, learner, data):
        self.estimators_ = []
        self.cls_vals = len(data.domain.class_var.values)
        for _ in range(learner.n_estimators):
            tree = Orange.classification.tree.SimpleTreeLearner(
                learner.min_instances, learner.max_depth, 
                learner.max_majority, learner.skip_prob, True)
            self.estimators_.append(tree(data))

    def predict_storage(self, data):
        p = np.zeros((data.X.shape[0], self.cls_vals))
        for tree in self.estimators_:
            p += tree(data, tree.Probs)
        p /= len(self.estimators_)
        return p.argmax(axis=1), p

if __name__ == '__main__':
    import time
    from Orange.data import Table
    d = Table('iris')

    l = RandomForestLearner(n_estimators=10, skip_prob='sqrt')
    m = l(d)
