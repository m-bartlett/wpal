from image import ANSI, np

def kmeans(pixels, iterations=3):

  # For this usage of k-means, k = 8 always as there are 8 ANSI colors to create clusters for

  cluster_centers = ANSI.copy()

  for i in range(iterations):

    # calculate distance between each pixel and each ANSI color
    distances = ( (cluster_centers - pixels[:, np.newaxis])**2 ).sum(axis=2)

    # Return indices of each ANSI color's closest neighbor pixel
    indices_of_closest_pixel_to_each_ANSI = np.argmin(distances, axis=0)

    # Substitute the actual rgb-arrays for the corresponding pixel index
    closest_pixel_to_each_ANSI = pixels[indices_of_closest_pixel_to_each_ANSI]

    closest_ANSI_index_per_pixel = np.argmin(distances, axis=1 )

    # Prevent not having any pixels being "closest" to a particular ANSI color
    pixels = np.concatenate([pixels, pixels[indices_of_closest_pixel_to_each_ANSI] ] )
    closest_ANSI_index_per_pixel = (
      np.concatenate([closest_ANSI_index_per_pixel, np.arange(ANSI.shape[0]) ] )
    )

    average_of_each_ANSIs_closest_neighbors = (
      np.array([ pixels[closest_ANSI_index_per_pixel==i].mean(axis=0) for i in range(ANSI.shape[0]) ])
    )

    cluster_centers = average_of_each_ANSIs_closest_neighbors

  ANSI_index, ANSI_nearest_neighbor_quantity = np.unique(closest_ANSI_index_per_pixel, return_counts = True)
  ANSI_indices_sorted_by_neighbor_quantity = np.flip(ANSI_nearest_neighbor_quantity[1:-1].argsort()+1)

  return cluster_centers, ANSI_indices_sorted_by_neighbor_quantity