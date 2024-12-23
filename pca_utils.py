import numpy as np
from eigen_utils import eig

def PCA(data_matrix, n_components):
    """
    Perform Principal Component Analysis (PCA) on the given data matrix.

    Parameters:
        data_matrix (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
                                     Each row corresponds to a data sample, and each column corresponds to a feature.
        n_components (int): The number of principal components to retain.

    Returns:
        tuple:
            reduced_data (numpy.ndarray): The data projected onto the top `n_components` principal components.
                                         Shape: (n_samples, n_components).
            reconstructed_data (numpy.ndarray): The data reconstructed back from the reduced representation.
                                               Shape: (n_samples, n_features).
    """
    # Compute the mean vector (mean of each feature across all samples)
    mean_vector = np.mean(data_matrix, axis=0)

    # Center the data matrix by subtracting the mean vector from each sample
    centered_matrix = data_matrix - mean_vector

    # Compute the covariance matrix of the centered data
    # Formula: Covariance = (1 / (n_samples - 1)) * (centered_matrix.T @ centered_matrix)
    covariance_matrix = np.dot(centered_matrix.T, centered_matrix) / (data_matrix.shape[0] - 1)

    # Compute eigenvalues and eigenvectors of the covariance matrix using a custom utility function
    eigvals, eigvecs = eig(covariance_matrix)

    # Sort the eigenvalues in descending order and get the sorted indices
    sorted_indices = np.argsort(eigvals)[::-1]

    # Select the indices of the top `n_components` eigenvalues
    top_indices = sorted_indices[:n_components]

    # Form the subset of eigenvectors corresponding to the top eigenvalues
    # These eigenvectors define the principal components
    eigenvector_subset = eigvecs[:, top_indices]

    # Project the centered data onto the top principal components
    reduced_data = np.dot(centered_matrix, eigenvector_subset)

    # Reconstruct the data from the reduced representation
    # Add the mean vector back to the reconstructed data
    reconstructed_data = np.dot(reduced_data, eigenvector_subset.T) + mean_vector

    return reduced_data, reconstructed_data
