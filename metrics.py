import pandas as pd
import numpy as np
from model import NearestNeighborBalled as NearestNeighbor

class Oversampled:
    def __init__(self, random_state = None, n_sample = 100):
        self._random_gen = np.random.RandomState(random_state)
        self.n_sample = n_sample
        self._synthethic_sample = []
        self._y_synthetic = []
    
    def resampled(self, X, y, minority):
        X_minor = np.asarray(X[y == minority])
        y_minor = np.asarray(y[y == minority])
        
        for _ in range(self.n_sample):
            idx = self._random_gen.randint(len(X_minor))
            self._y_synthetic.append(y_minor[idx])
            self._synthethic_sample.append(X_minor[idx])
        
        X_synthetic = pd.DataFrame(self._synthethic_sample, columns=X.columns)
        y_synthetic = pd.Series(self._y_synthetic, name=y.name)

        X_resampled = pd.concat([X, X_synthetic]).reset_index(drop=True)
        y_resampled = pd.concat([y, y_synthetic]).reset_index(drop=True)

        return X_resampled, y_resampled

def train_test_split(X, y, train_size, random_state=None,*,stratisfy=None):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('X Datatype is not a DataFrame')
    if not isinstance(y, pd.Series):
        raise ValueError('Y Datatype is not a Series')
    
    train = {'X': [], 'y': []}
    test = {'X': [], 'y': []}

    np.random.seed(random_state)

    if stratisfy is None:
        train_size = int(len(X) * train_size)
        idx = np.random.permutation(len(X))
        
        train['X'], train['y'] = X.iloc[idx[:train_size]], y.iloc[idx[:train_size]]
        test['X'], test['y'] = X.iloc[idx[train_size:]], y.iloc[idx[train_size:]]
    else:
        for labels in np.unique(stratisfy):
            i = y[y == labels].index
            x_i = X.loc[i]
            y_i = y.loc[i]
            idx = np.random.permutation(len(x_i))
            train_i = int(len(x_i) * train_size)
            
            train['X'].append(x_i.iloc[idx[:train_i]])
            train['y'].append(y_i.iloc[idx[:train_i]])
            test['X'].append(x_i.iloc[idx[train_i:]])
            test['y'].append(y_i.iloc[idx[train_i:]])

        for x in train:
            train[x] = pd.concat(train[x])
            test[x] = pd.concat(test[x])
    
    return train['X'].reset_index(drop=True), test['X'].reset_index(drop=True), train['y'].reset_index(drop=True), test['y'].reset_index(drop=True)

class SMOTE:
    def __init__(self, n_neighbors = 5, random_state=None, n_sample=100, leaf_node=30, p=3, metrics='euclidean'):
        self._random_gen = np.random.RandomState(random_state)
        self.n_sample = n_sample
        self.k = n_neighbors
        self._p = p
        self._leaf_node = leaf_node
        self._metrics = metrics
        self._synthethic_sample = []
        self._y_synthetic = []
    
    def resampled(self, X, y, minority):
        X_minor = np.asarray(X[y == minority])
        y_minor = np.asarray(y[y == minority])
        
        ballnn = NearestNeighbor(n_neighbors=self.k, p=self._p, leaf_node=self._leaf_node, metrics=self._metrics)
        ballnn.fit(X_minor)
        
        for _ in range(self.n_sample):
            idx = self._random_gen.randint(0, len(X_minor))
            x_i = X_minor[idx]
            self._y_synthetic.append(y_minor[idx])
            
            nearest = ballnn.kneighbors(x_i)
            idx_neighbor = nearest[0][0]
            x_neighbor = X_minor[idx_neighbor]
            
            lambda_ = self._random_gen.rand()
            synthetic_sample = x_i + lambda_ * (x_neighbor - x_i)
            self._synthethic_sample.append(synthetic_sample)
            
        X_synthetic = pd.DataFrame(self._synthethic_sample, columns=X.columns)
        y_synthetic = pd.Series(self._y_synthetic, name=y.name)
        
        X_resampled = pd.concat([X, X_synthetic]).reset_index(drop=True)
        y_resampled = pd.concat([y, y_synthetic]).reset_index(drop=True)
        return X_resampled, y_resampled

def StandardScaler(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def MinMaxScaler(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))