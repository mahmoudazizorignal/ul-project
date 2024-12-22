import numpy as np

def eig(matrix, num_iterations=1000, tol=1e-10):
    """
    Compute all eigenvalues and eigenvectors of a matrix using the QR Algorithm.

    Args:
        matrix (numpy.ndarray): A square matrix (n x n).
        num_iterations (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        tuple: Eigenvalues (diagonal elements) and eigenvectors (columns of Q).
    """
    n = matrix.shape[0]
    A = np.array(matrix, dtype=np.float64)  # Ensure double precision
    Q_total = np.eye(n)  # Initialize Q_total to store cumulative Q

    for _ in range(num_iterations):
        # QR decomposition
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)  # Update A
        
        # Accumulate the eigenvectors
        Q_total = np.dot(Q_total, Q)
        
        # Check for convergence (off-diagonal elements close to zero)
        if np.allclose(A - np.diag(np.diagonal(A)), 0, atol=tol):
            break
    
    eigenvalues = np.diagonal(A)
    eigenvectors = Q_total
    return eigenvalues, eigenvectors





if __name__ == "__main__":
    mat = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ], dtype=float)

    eig = eig(mat, num_iterations=1000, tol=1e-8)
    print(eig)
    
    eigvectors = np.linalg.eig(mat)
    print(eigvectors)


    # eigen_vectors = eigenvectors(mat)
    # print(eigen_vectors)
    