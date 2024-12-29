import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class FuzzyCMeans:
    """
    Fuzzy C-Means clustering algorithm implementation.
    
    Attributes:
        n_clusters (int): Number of clusters
        max_iter (int): Maximum number of iterations
        m (float): Fuzziness parameter (m > 1)
        epsilon (float): Convergence threshold
        centers (np.ndarray): Cluster centers
        membership_matrix (np.ndarray): Fuzzy membership matrix
    """
    
    def __init__(self, n_clusters: int = 2, max_iter: int = 100, 
                 m: float = 2.0, epsilon: float = 1e-5):
        """
        Initialize the Fuzzy C-Means clustering algorithm.
        
        Args:
            n_clusters: Number of clusters (default: 2)
            max_iter: Maximum number of iterations (default: 100)
            m: Fuzziness parameter, must be > 1 (default: 2.0)
            epsilon: Convergence threshold (default: 1e-5)
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.epsilon = epsilon
        self.centers = None
        self.membership_matrix = None
        
    def _initialize_membership_matrix(self, n_samples: int) -> np.ndarray:
        """
        Initialize the membership matrix with random values.
        
        Args:
            n_samples: Number of samples in the dataset
            
        Returns:
            Initialized membership matrix
        """
        membership_matrix = np.random.rand(n_samples, self.n_clusters)
        # Normalize the matrix so that the sum of memberships for each sample equals 1
        return membership_matrix / membership_matrix.sum(axis=1, keepdims=True)
    
    def _update_centers(self, X: np.ndarray, membership_matrix: np.ndarray) -> np.ndarray:
        """
        Update cluster centers based on membership matrix.
        
        Args:
            X: Input data
            membership_matrix: Current membership matrix
            
        Returns:
            Updated cluster centers
        """
        weights = membership_matrix ** self.m
        return (X.T @ weights) / weights.sum(axis=0)
    
    def _update_membership_matrix(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Update membership matrix based on current centers.
        
        Args:
            X: Input data
            centers: Current cluster centers
            
        Returns:
            Updated membership matrix
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - centers[i], axis=1)
        
        # Handle division by zero
        distances = np.fmax(distances, np.finfo(float).eps)
        
        exp = 2.0 / (self.m - 1)
        distances = distances ** (-exp)
        return distances / distances.sum(axis=1, keepdims=True)
    
    def fit(self, X: np.ndarray) -> 'FuzzyCMeans':
        """
        Fit the Fuzzy C-Means model to the input data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            self: The fitted model
        """
        n_samples = X.shape[0]
        
        # Initialize membership matrix
        self.membership_matrix = self._initialize_membership_matrix(n_samples)
        
        for _ in range(self.max_iter):
            # Store old membership matrix
            old_membership_matrix = self.membership_matrix.copy()
            
            # Update centers and membership matrix
            self.centers = self._update_centers(X, self.membership_matrix)
            self.membership_matrix = self._update_membership_matrix(X, self.centers)
            
            # Check convergence
            if np.linalg.norm(self.membership_matrix - old_membership_matrix) < self.epsilon:
                break
                
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster membership for new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels for each sample
        """
        if self.centers is None:
            raise ValueError("Model must be fitted before making predictions")
            
        membership_matrix = self._update_membership_matrix(X, self.centers)
        return np.argmax(membership_matrix, axis=1)

def plot_clusters(X: np.ndarray, labels: np.ndarray, centers: Optional[np.ndarray] = None):
    """
    Plot the clustering results.
    
    Args:
        X: Input data
        labels: Cluster labels
        centers: Cluster centers (optional)
    """
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidth=3)
    plt.title('Fuzzy C-Means Clustering Results')
    plt.show()


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (100, 2)),
        np.random.normal(4, 1, (100, 2))
    ])
    
    # Create and fit the model
    fcm = FuzzyCMeans(n_clusters=2)
    fcm.fit(X)
    
    # Make predictions
    labels = fcm.predict(X)
    
    # Plot results
    plot_clusters(X, labels, fcm.centers)