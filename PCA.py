def PCA(data_matrix, n_components):
    mean_vector = np.mean(data_matrix, axis=0)
    centered_matrix = data_matrix - mean_vector
    covariance_matrix = np.dot(centered_matrix.T, centered_matrix) / (data_matrix.shape[0] - 1)
    eigvals, eigvecs = eig(covariance_matrix)
    sorted_indices = np.argsort(eigvals)[::-1]
    top_indices = sorted_indices[:n_components]
    
    eigenvector_subset = eigvecs[:, top_indices]
    reduced_data = np.dot(centered_matrix, eigenvector_subset)
    reconstructed_data = np.dot(reduced_data, eigenvector_subset.T) + mean_vector
    
    return reduced_data, reconstructed_data