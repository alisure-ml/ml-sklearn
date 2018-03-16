# knn
k-Nearest Neighbor,kd-tree, ball-tree,

###
The [sklearn.neighbors](http://scikit-learn.org/dev/modules/classes.html#module-sklearn.neighbors) 
module implements the k-nearest neighbors algorithm.
> http://scikit-learn.org/dev/modules/neighbors.html#neighbors

> sklearn.neighbors provides functionality for **unsupervised and supervised 
neighbors-based learning methods**, notably manifold learning and spectral 
clustering.
* Supervised neighbors-based learning comes in two flavors:  
    - classification for data with discrete labels,  
    - regression for data with continuous labels.  
    
* The principle behind nearest neighbor method is to find a predefined number of training 
samples closest in distance to the new point, and predict the label from these.

* The number of samples can be a user-defined constant (k-nearest neighbor learning), 
or vary based on the local density of points (radius-distance neighbor learning).

* The distance be any metric measure: standard Euclidean distance is the most common choice.

* Neighbor-based methods are know as non-generalizing machine learning methods, since they 
simply "remember" all of its training data (possibly transformed into a fast indexing 
structure such as a Ball Tree or KD Tree).


