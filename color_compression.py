from skimage import io
from sklearn.cluster import KMeans
import numpy as np


# Read the image
img = io.imread('original_image.png')
io.imshow(img)
io.show()

# Remove alpha channel

img = img[:, :, :3]

# Dimension of the original image
rows = img.shape[0]
cols = img.shape[1]

# Flatten the image
img = img.reshape(rows * cols, 3)

# Implement k-means clustering to form k clusters
kmeans = KMeans(n_clusters=8)
kmeans.fit(img)

# Replace each pixel value with its nearby centroid
compressed_img = kmeans.cluster_centers_[kmeans.labels_]
compressed_img = np.clip(compressed_img.astype('uint8'), 0, 255)

# Reshape the image to original dimension
compressed_img = compressed_img.reshape(rows, cols, 3)

# Save and display output image
io.imsave('compressed_image_8.png', compressed_img)
io.imshow(compressed_img)
io.show()