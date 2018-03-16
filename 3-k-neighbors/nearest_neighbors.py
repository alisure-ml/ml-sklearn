from sklearn import neighbors
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
n_brs = neighbors.NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
distances, indices = n_brs.kneighbors(X)
print(distances)
print(indices)

sparse_graph = n_brs.kneighbors_graph(X).toarray()
print(sparse_graph)
