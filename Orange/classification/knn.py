import sklearn.neighbors as skl_neighbors
import sklearn.preprocessing as skl_preprocessing
from numpy import isnan, cov
import Orange.data
from Orange.classification import Learner, Model
from Orange.data.continuizer import DomainContinuizer

__all__ = ["KNNLearner"]


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)

def replace_nan(X, imp_model):
        # Default scikit Imputer
        # Use Orange imputer when implemented
        if isnan(X).sum():
                X = imp_model.transform(X)
        return X


class KNNLearner(Learner):
    def __init__(self, n_neighbors=5, metric="euclidean", normalize=True,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.normalize = normalize

    def _domain_continuizer(self, data):
        multinomial = continuous = None
        if any(map(is_discrete, data.domain.attributes)):
            multinomial = DomainContinuizer.FrequentIsBase
        if self.normalize:
            continuous = DomainContinuizer.NormalizeBySD
        if multinomial is not None or continuous is not None:
            return DomainContinuizer(multinomial_treatment=multinomial,
                                     normalize_continuous=continuous)
        else:
            return None

    def __call__(self, data):
        dc = self._domain_continuizer(data)
        if dc is not None:
            domain = dc(data)
            data = Orange.data.Table.from_table(domain, data)

        return super().__call__(data)

    def fit(self, X, Y, W):
        self.imputer = skl_preprocessing.Imputer()
        self.imputer.fit(X)
        X = replace_nan(X, self.imputer)
        if self.metric == "mahalanobis":
            skclf = skl_neighbors.KNeighborsClassifier(
                n_neighbors=self.n_neighbors, metric=self.metric, V = cov(X.T)
            )
        else:
            skclf = skl_neighbors.KNeighborsClassifier(
                n_neighbors=self.n_neighbors, metric=self.metric
            )
        skclf.fit(X, Y.ravel())
        return KNNModel(skclf, self.imputer)


class KNNModel(Model):
    def __init__(self, clf, imp):
        self.clf = clf
        self.imputer = imp

    def predict(self, X):
        X = replace_nan(X, imp_model=self.imputer)
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob
