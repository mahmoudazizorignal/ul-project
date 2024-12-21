import numpy as np
from sympy import symbols, solve, det, Matrix

def eigenvals(A: np.ndarray) -> np.ndarray:
    """
    Finds the eigen values of the matrix.
    
    Args:
        A (np.ndarray): the matrix to find its eigen values.
        
    Returns:
        (np.ndarray): the eigenvalues of the matrix.
    """
    n = mat.shape[0]
    λ = symbols("λ")
    A_sym = Matrix(A)
    identity_matrix = Matrix(np.eye(n))
    char_poly = det(A_sym - λ * identity_matrix)
    eigen_values = solve(char_poly, λ)
    return np.array(eigen_values, dtype=float)


def nullspace(A: np.ndarray) -> np.ndarray:
    """
    Finds the nullspace of a matrix.

    Args:
        A (np.ndarray): the matrix to find its nullspace.
    
    Returns:
        np.ndarray: the basis of the nullspace.
    """
    sympy_matrix = Matrix(A)
    null_basis = sympy_matrix.nullspace()
    return np.array(null_basis, dtype=float)

def eigenvectors(A: np.ndarray) -> np.ndarray:
    """
    Finds the eigenvectors of the matrix.

    Args:
        A (np.ndarray): the matrix to find its eigenvectors.

    Returns:
        np.ndarray: the eigenvectors of the matrix.
    """
    eigen_values = eigenvals(A)
    eigen_vectors = None
    identity_matrix = np.eye(A.shape[0])
    for eigen_value in eigen_values:
        M = A - eigen_value * identity_matrix
        null_basis = nullspace(M)
        null_basis = null_basis.reshape(null_basis.shape[0], null_basis.shape[1]).T
        if eigen_vectors is None:
            eigen_vectors = null_basis
        else:
            eigen_vectors = np.append(eigen_vectors, null_basis, axis=1)
    return eigen_vectors


if __name__ == "__main__":
    mat = np.array([
        [6, 10, 6],
        [0, 8, 12],
        [0, 0, 2]
    ], dtype=float)

    eigen_values = eigenvals(mat)
    print(eigen_values)


    eigen_vectors = eigenvectors(mat)
    print(eigen_vectors)
    