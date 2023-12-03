import numpy as np
from numba import jit


@jit(nopython=True, parallel=True)
def nmf_hs_l1_2(X, delta, lambd, iters, components):
    (rows, cols) = X.shape
    # Initialize A and S
    A = np.random.rand(rows, components)
    S = np.random.rand(components, cols)
    for i in range(cols):
        magn = np.linalg.norm(S[:,i])
        S[:,i] = S[:,i] / magn

    all_delta_X = delta * np.ones((1, cols))
    all_delta_A = delta * np.ones((1, components))
    X_f = np.vstack((X, all_delta_X))
    for _ in range(0, iters):
        A_f = np.vstack((A, all_delta_A))
        A = A * (X @ np.transpose(S)) / (A @ S @ np.transpose(S))
        S = S * (np.transpose(A_f) @ X_f) / (np.transpose(A_f) @ A_f @ S + (lambd / 2) * np.sum(np.sqrt(S + 1e-6)))
    error = (np.linalg.norm(X - A @ S))
    return A, S, error 