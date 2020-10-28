from PIL import Image
import numpy as np
from copy import deepcopy


def kmeans(img_1d, k_clusters, max_iter, init_centroids='random'):
  '''
  Inputs:
    img_1d : np.ndarray with shape=(height * width, num_channels)
      Original image in 1d array
    
    k_clusters : int
      Number of clusters

    max_iter : int
      Max iterator

    init_centroids : str
      The way which use to init centroids
      'random' --> centroid has `c` channels, with `c` is initial random in [0,255]
      'in_pixels' --> centroid is a random pixels of original image

  Outputs:
    centroids : np.ndarray with shape=(k_clusters, num_channels)
      Store color centroids

    labels : np.ndarray with shape=(height * width, )
      Store label for pixels (cluster's index on which the pixel belongs)
  '''
  # Number of data
  num_data = img_1d.shape[0]
  
  # Number of features of data
  num_features = img_1d.shape[1]
  
  # Generate centroids for every clusters
  if init_centroids == 'random':
    centroids = np.random.randint(256, size=(k_clusters, num_features))
  else: # in_pixels
    centroids = np.array([img_1d[np.random.randint(num_data)] for _ in range(k_clusters)])
    
  # to tell which cluster that pixel belong
  labels = np.zeros(num_data)
  
  # to store the distance from 1 pixel to each current centroid
  distances = np.zeros((num_data, k_clusters)) 

  # Begin of k-means algorithm
  diff = 1 # just not 0
  it = 0
  
  while diff != 0 and it < max_iter:
    prev_centroids = deepcopy(centroids)
    
    # Measure the distance from 1 pixel to every current centroids
    for i in range(k_clusters):
      distances[:, i] = np.linalg.norm(img_1d - centroids[i], axis=1)
      
    # Assign all pixel to its closest current centroid
    labels = np.argmin(distances, axis=1)
    
    # Repositioning every centroids
    for i in range(k_clusters):
      centroids[i] = np.mean(img_1d[labels == i], axis=0)
    
    diff = np.linalg.norm(centroids - prev_centroids)
    it += 1
  
  return centroids, labels


if __name__ == '__main__':
  # Load an image
  img = Image.open('original_image.jpg')
  width, height = img.size
  
  # Change the image to 1d array
  img = np.reshape(img, (height * width, 3))
        
  for k in [ 3, 5, 7, 20 ]:
    print('Running k-means with k = ' + str(k) + " ...")
    
    centroids, labels = kmeans(img, k_clusters=k, max_iter=100, init_centroids='random')
    
    compressed_img = deepcopy(img)
    
    for i in range(compressed_img.shape[0]):
      compressed_img[i] = centroids[ labels[i] ]
    
    # Change the image to the original shape
    compressed_img = np.reshape(compressed_img, (height, width, 3))
    compressed_img = Image.fromarray(compressed_img)
    compressed_img.save('compressed_image_' + str(k) + '.jpg')