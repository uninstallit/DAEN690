import numpy as np

from sklearn.neighbors import BallTree



# source: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html

rng = np.random.RandomState(0)

# Nearest by K
X = rng.random_sample((10, 2))  # 10 points in 3 dimensions
X = np.array(X)
print(f'X:{X}')
tree = BallTree(X, metric='haversine')   
K=3           
dist, ind = tree.query(X[:1], k=K)    
print(f'* Query item :{X[:1]}')
print(f'By distance closest neighors K={3} - Found indices:{ind}, distance closest neighors {dist}')  # indices of 3 closest neighbors

# Nearest by Radius
import numpy as np
rng = np.random.RandomState(0)
X = rng.random_sample((10, 2))  # 10 points in 3 dimensions
tree = BallTree(X, leaf_size=2)  
radius = 0.3    
print(f'* Query item:{X[:1]}')
# print(tree.query_radius(X[:1], r=radius, count_only=True))
ind = tree.query_radius(X[:1], r=radius)  
print(f'By closet neighbors within radius distance={radius}, Found indices:{ind}')


