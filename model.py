import numpy as np
import json
import os

class KNeighborsClassfier:
    def __init__(self, n_neighbors = 3, p = 3, method = 'auto', leaf_node=30, metrics = 'euclidean'):
        if method not in ['brute', 'kd_tree', 'ball_tree', 'auto']:
            raise ValueError('Unsupported Method use brute, kd_tree, ball_tree or auto')
        
        if metrics not in ['euclidean', 'manhattan', 'minkowski']:
            raise ValueError('Unsupported Distance Metrics use euclidean, manhattan or minkowski')

        self._k = n_neighbors
        self._p = p
        self._method = method
        self._metrics = metrics
        self._leaf_node = leaf_node
        self._tree = None
        self._X = None
        self._y = None
    
    def fit(self, X, y):
        self._X = np.asarray(X)
        self._y = np.asarray(y)
        n_sample, n_feature = X.shape

        if self._method == 'auto':
            if n_sample > 100:
                self._method = 'ball_tree'
            elif n_feature > 20:
                self._method = 'kd_tree'
            else:
                self._method = 'brute'

        if self._method == 'kd_tree':
            self._tree = self._build_kd_tree(self._X, np.arange(len(X)))
        elif self._method == 'ball_tree':
            self._tree = self._build_ball_tree(self._X, np.arange(len(X)))

    def _build_kd_tree(self, points, index, depth = 0):
        if len(points) <= self._leaf_node:
            return {'leaf': True, 'points': points, 'index': index}
        
        axis = depth % points.shape[1]
        sorted_idx = points[:, axis].argsort()
        points = points[sorted_idx]
        index = index[sorted_idx]
        median_idx = len(points) // 2

        return {
            'leaf': False,
            'point': points[median_idx],
            'axis': axis,
            'left': self._build_kd_tree(points[:median_idx], index[:median_idx], depth + 1),
            'right': self._build_kd_tree(points[median_idx + 1:], index[median_idx + 1:], depth + 1)
        }
    
    def _build_ball_tree(self, points, index):
        if len(points) <= self._leaf_node:
            return {'leaf': True, 'points': points, 'index': index}
        
        centroid = np.mean(points, axis=0)
        distance = self._distance(centroid, points)
        sorted_idx = distance.argsort()
        radius = distance[sorted_idx]
        points = points[sorted_idx]
        index = index[sorted_idx]
        median_idx = len(points) // 2

        return {
            'leaf': False,
            'centroid': centroid,
            'radius': radius[median_idx],
            'left': self._build_ball_tree(points[:median_idx], index[:median_idx]),
            'right': self._build_ball_tree(points[median_idx + 1:], index[median_idx + 1:])
        }
        

    def _distance(self, p, q, axis = None):
        if axis is None:
            axis = 1 if q.ndim > 1 else 0

        if self._metrics == 'euclidean':
            return np.sqrt(np.sum((p - q) ** 2, axis=axis))
        elif self._metrics == 'manhattan':
            return np.sum(np.abs(p - q), axis=axis)
        elif self._metrics == 'minkowski':
            return np.power(np.sum((p - q) ** self._p, axis=axis), 1/self._p)
            
    def predict(self, X):
        X = np.asarray(X)
        X = np.atleast_2d(X)

        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        if self._method == 'brute':
            distance = self._distance(self._X, x)
            neighbor_idx = distance.argsort()[:self._k]
        elif self._method == 'kd_tree':
            neighbor_idx = self._query_kd_tree(self._tree, x)[0]
        elif self._method == 'ball_tree':
            neighbor_idx = self._query_ball_tree(self._tree, x)[0]
                    
        neighbor_labels = self._y[neighbor_idx]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        return labels[counts.argmax()]

    def _query_kd_tree(self, node, query):
        if node['leaf']:
            distance = self._distance(query, node['points'])
            return node['index'], distance
        
        axis = node['axis']
        first, second = ('left', 'right') if query[axis] < node['point'][axis] else ('right', 'left')
        
        nearest_idx, nearest_dist = self._query_kd_tree(node[first], query)

        if len(nearest_idx) < self._k or abs(query[axis] - node['point'][axis]) < np.max(nearest_dist):
            opposite_idx, opposite_dist = self._query_kd_tree(node[second], query)

            combined_idx = np.concatenate([nearest_idx, opposite_idx])
            combined_dist = np.concatenate([nearest_dist, opposite_dist])

            if len(combined_idx) > self._k:
                idx = combined_dist.argsort()[:self._k]
                nearest_idx = combined_idx[idx]
                nearest_dist = combined_dist[idx]
            else:
                nearest_idx = combined_idx
                nearest_dist = combined_dist
        
        return nearest_idx, nearest_dist
    
    def _query_ball_tree(self, node, query):
        if node['leaf']:
            distance = self._distance(query, node['points'])
            return node['index'], distance
        
        dist_to_centroid = self._distance(node['centroid'], query)

        first, second = ('left', 'right') if dist_to_centroid < node['radius'] else ('right', 'left')

        nearest_idx, nearest_dist = self._query_ball_tree(node[first], query)

        if len(nearest_idx) < self._k or abs(dist_to_centroid - node['radius']) < np.max(nearest_dist) if len(nearest_dist) > 0 else True:
            opposite_idx, opposite_dist = self._query_ball_tree(node[second], query)

            combined_idx = np.concatenate([nearest_idx, opposite_idx])
            combined_dist = np.concatenate([nearest_dist, opposite_dist])

            if len(combined_idx) > self._k:
                idx = combined_dist.argsort()[:self._k]
                nearest_idx = combined_idx[idx]
                nearest_dist = combined_dist[idx]
            else:
                nearest_idx = combined_idx
                nearest_dist = combined_dist
        
        return nearest_idx, nearest_dist
    
    def save_model(self, file_name):
        data = {
            'n_neighbors': self._k,
            'metrics': self._metrics,
            'method': self._method,
            'leaf_node': self._leaf_node,
            'p': self._p,
            'X_train': self._X.tolist(),
            'y_train': self._y.tolist(),
            'tree': self._serialized(self._tree)
        }
        
        path = os.getcwd()
        os.makedirs(os.path.join(path, 'model'), exist_ok=True)
        path = os.path.join(path, 'model', file_name)
        
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        print('Model has been successfully saved!')
    
    def _serialized(self, node):
        if self._method == 'kd_tree':
            if node['leaf']:
                return {'leaf': True, 'points': node['points'].tolist(), 'index': node['index'].tolist()}
            
            return {
                'leaf': False,
                'point': float(node['point']),
                'axis': int(node['axis']),
                'left': self._serialized(node['left']),
                'right': self._serialized(node['right'])
            }
            
        elif self._method == 'ball_tree':
            if node['leaf']:
                return {'leaf': True, 'points': node['points'].tolist(), 'index': node['index'].tolist()}
            
            return {
                'leaf': False,
                'centroid': node['centroid'].tolist(),
                'radius': float(node['radius']),
                'left': self._serialized(node['left']),
                'right': self._serialized(node['right'])
            }
        else:
            return None
            
    def load_model(self, file_name):
        path = os.getcwd()
        
        try:
            path = os.path.join(path, 'model', file_name)
        except FileNotFoundError:
            raise ValueError('File or Folder Model not Found!')
        
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        
        self._k = int(data['n_neighbors']),
        self._method = str(data['method']),
        self._k = self._k[0]
        self._method = self._method[0]
        self._metrics = data['metrics']
        self._leaf_node = data['leaf_node']
        self._p = data['p']
        self._X = np.array(data['X_train'])
        self._y = np.array(data['y_train'])
        self._tree = self._deserialized(data['tree'])
      
        print('Model has been loaded!')
        
    def _deserialized(self, node):
        if self._method == 'kd_tree':
            if node['leaf']:
                return {'leaf': True, 'points': np.array(node['points']), 'index': np.array(node['index'])}
            
            return {
                'leaf': False,
                'point': np.float64(node['point']),
                'axis': np.int32(node['axis']),
                'left': self._deserialized(node['left']),
                'right': self._deserialized(node['right'])
            }
        elif self._method == 'ball_tree':
            if node['leaf']:
                return {'leaf': True, 'points': np.array(node['points']), 'index': np.array(node['index'])}
            
            return {
                'leaf': False,
                'centroid': np.array(node['centroid']),
                'radius': np.float64(node['radius']),
                'left': self._deserialized(node['left']),
                'right': self._deserialized(node['right'])
            }
        else:
            return None
        
class NearestNeighborBalled:
    def __init__(self, n_neighbors = 5, p=3, leaf_node=30, metrics='euclidean'):
        if metrics not in ['euclidean', 'manhattan', 'minkowski', 'chebyshev']:
            raise ValueError('Unsupported Metrics, use euclidean, manhattan, minkowski or chebyshev instead')
        
        self._metrics = metrics
        self.k = n_neighbors
        self.leaf_node = leaf_node
        self.p = p
        self._tree = None
    
    def fit(self, X):
        self.X = np.asarray(X)
        
        self._tree =  self._build_ball_tree(self.X, np.arange(len(self.X)))
        
    def _build_ball_tree(self, points, index):
        if len(points) <= self.leaf_node:
            return {'leaf': True, 'points': points, 'index': index}
        
        centroid = np.mean(points, axis=0)
        radius = self._distance(centroid, points)
        sorted_idx = radius.argsort()
        radius = radius[sorted_idx]
        points = points[sorted_idx]
        index = index[sorted_idx]
        median_idx = len(points) // 2
        
        return {
            'leaf': False,
            'centroid': centroid,
            'radius': radius[median_idx],
            'left': self._build_ball_tree(points[:median_idx], index[:median_idx]),
            'right': self._build_ball_tree(points[median_idx + 1:], index[median_idx + 1:])
        }
    
    def kneighbors(self, X):
        X = np.asarray(X)
        X = np.atleast_2d(X)
        
        return np.array([self._query(self._tree, x)[0] for x in X])
    
    def _query(self, node, query):
        if node['leaf']:
            distance = self._distance(node['points'], query, axis=1)
            return node['index'], distance
        
        dist_to_centroid = self._distance(node['centroid'], query)
        first, second = ('left', 'right') if dist_to_centroid < node['radius'] else ('right', 'left')
        
        nearest_idx, nearest_dist = self._query(node[first], query)
                
        if len(nearest_idx) < self.k or abs(dist_to_centroid - node['radius']) < np.max(nearest_dist) if len(nearest_dist) > 0 else True:
            opposite_idx, opposite_dist = self._query(node[second], query)
            
            combined_idx = np.concatenate([nearest_idx, opposite_idx])
            combined_dist = np.concatenate([nearest_dist, opposite_dist])
            
            
            if len(combined_idx) > self.k:
                idx = combined_dist.argsort()[:self.k]
                nearest_idx = combined_idx[idx]
                nearest_dist = combined_dist[idx]
            else:
                nearest_idx = combined_idx
                nearest_dist = combined_dist
                
        return nearest_idx, nearest_dist
        
    def _distance(self, x1, x2, axis=None):
        if axis is None:
            axis = 1 if x2.ndim > 1 else 0
            
        if self._metrics == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=axis))
        elif self._metrics == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=axis)
        elif self._metrics == 'chebyshev':
            return np.max(np.abs(x1 - x2), axis=axis)
        elif self._metrics == 'minkowski':
            return np.power(np.sum((x1 - x2) ** self.p, axis=axis), 1/self.p)