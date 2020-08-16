from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt


# Generate 3 sets of data to surround each predefined center
dummy_center_1 = np.array([1, 1])
dummy_center_2 = np.array([5, 5])
dummy_center_3 = np.array([8, 1])

data_1 = np.random.randn(300, 2) + dummy_center_1
data_2 = np.random.randn(300, 2) + dummy_center_2
data_3 = np.random.randn(300, 2) + dummy_center_3

data = np.concatenate((data_1, data_2, data_3), axis=0)

plt.scatter(data[:, 0], data[:, 1], s=5)

# Number of clusters
num_clusters = 3

# Number of data
num_data = data.shape[0]

# Number of features of data
num_features = data.shape[1]

# Generate 3 random centers of each cluster
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
centers = np.random.randn(num_clusters, num_features) * std + mean

plt.scatter(centers[:, 0], centers[:, 1], s=170, c='g', marker='*')

# Begin of k-means algorithm
clusters = np.zeros(num_data) # store which cluster 1 point currently in
distances = np.zeros((num_data, num_clusters)) # store distance from 1 point to each current center

diff = 1 # difference between new and old centers

while diff != 0:
    prev_centers = deepcopy(centers)
  
    # Measure the distance from 1 point to every current center
    for i in range(num_clusters):
        distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
    
    # Assign all data to its closest current center
    clusters = np.argmin(distances, axis=1)
    
    for i in range(num_clusters):
        centers[i] = np.mean(data[clusters == i], axis=0)
        
    diff = np.linalg.norm(centers - prev_centers)
    
plt.scatter(centers[:, 0], centers[:, 1], s=170, c='r', marker='*')