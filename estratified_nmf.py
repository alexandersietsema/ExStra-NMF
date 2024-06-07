# stratified_nmf.py
"""
This file implements the Stratified-NMF algorithm. 
"""


import numpy as np
from scipy.sparse import csr_array
from tqdm import trange
import matplotlib.pyplot as plt
from termcolor import cprint


StratifiedNMFReturnType = tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]


def update_V(
    A: list[np.ndarray],
    V: list[np.ndarray],
    X: list[np.ndarray],
    W: list[np.ndarray],
    H: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Updates V using the multiplicative update rule."""
    strata = len(A)
    out = [np.zeros(V[s].shape) for s in range(strata)]

    for s in range(strata):
        rows = A[s].shape[0]
        
        num = A[s] @ X[s].T
        den = V[s] @ X[s] @ X[s].T + W[s] @ H @ X[s].T + tol        
        #den = V[s] * rows + H.T @ np.sum(W[s], axis=0) + tol
        #if isinstance(A[s], csr_array):
        #    out[s] = V[s] * A[s].sum(axis=0) / den
        #else:
        #    out[s] = V[s] * np.sum(A[s], axis=0) / den
        out[s] = V[s] * (num / den)
    return out


def update_X(
    A: list[np.ndarray],
    V: list[np.ndarray],
    X: list[np.ndarray],
    W: list[np.ndarray],
    H: np.ndarray,
    tol: float = 1e-9,
    reg: float = 2.0,
) -> np.ndarray:
    """Updates W using the multiplicative update rule."""
    strata = len(A)
    out = [np.zeros(X[s].shape) for s in range(strata)]

    for s in range(strata):
        rows = A[s].shape[0]
        
        num = V[s].T @ A[s]
        den = V[s].T @ V[s] @ X[s]+ V[s].T @ W[s] @ H + tol # + 10*X[s]
       
        den += reg * (X [s] > 0.0001)
        #den += reg*np.ones((X[s].shape[0], H.shape[0])) @ H
        #den += 1*(np.sum([np.ones((X[s].shape[0], X[s].shape[0])) @ X[i] for i in range(strata) if i != s]))
        
        #den = V[s] * rows + H.T @ np.sum(W[s], axis=0) + tol
        #if isinstance(A[s], csr_array):
        #    out[s] = V[s] * A[s].sum(axis=0) / den
        #else:
        #    out[s] = V[s] * np.sum(A[s], axis=0) / den
        out[s] = X[s] * (num / den)
        #print(s, num.mean(), den.mean(), (num/den).mean(), out[s].mean())
    return out

def update_W(
    A: list[np.ndarray],
    V: list[np.ndarray],
    X: list[np.ndarray],
    W: list[np.ndarray],
    H: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Updates W using the multiplicative update rule."""
    strata = len(A)
    out = [np.zeros(W[s].shape) for s in range(strata)]

    for s in range(strata):
        rows = A[s].shape[0]
        
        num = A[s] @ H.T
        den = V[s] @ X[s] @ H.T + W[s] @ H @ H.T + tol
        #den = V[s] * rows + H.T @ np.sum(W[s], axis=0) + tol
        #if isinstance(A[s], csr_array):
        #    out[s] = V[s] * A[s].sum(axis=0) / den
        #else:
        #    out[s] = V[s] * np.sum(A[s], axis=0) / den
        out[s] = W[s] * (num / den)
    return out

def update_H(
    A: list[np.ndarray],
    V: list[np.ndarray],
    X: list[np.ndarray],
    W: list[np.ndarray],
    H: np.ndarray,
    tol: float = 1e-9,
    reg: float = 2.0,
) -> np.ndarray:
    """Updates H using the multiplicative update rule."""
    strata = len(A)
    out = np.zeros(H.shape)

    num = 0
    den = 0

    for s in range(strata):
        
        num += W[s].T @ A[s]
        
        den += W[s].T @ V[s] @ X[s] + W[s].T @ W[s] @ H
        
        #den += reg * np.ones((H.shape[0], X[s].shape[0])) @ X[s]
        #if isinstance(A[s], csr_array):
        #    num += ((A[s].T).dot(W[s])).T
        #else:
        #    num += np.dot(W[s].T, A[s])
        #den += np.outer(np.sum(W[s], axis=0), V[s]) + np.dot(np.dot(W[s].T, W[s]), H)
    out = H * num / (den + tol) # + 10*h

    return out


def loss(
    A: list[np.ndarray],
    V: list[np.ndarray],
    X: list[np.ndarray],
    W: list[np.ndarray],
    H: np.ndarray,
) -> float:
    """Calculates the loss sqrt(sum_s ||A(s) - 1 v(s)^T - W(s) H||_F^2 )"""
    strata = len(A)
    out = 0.0
    for s in range(strata):
        rows = A[s].shape[0]

        out += (
            np.linalg.norm(A[s] - V[s] @ X[s] - W[s] @ H) ** 2
        )
    return out**0.5


def estratified_nmf(
    A: list[np.ndarray],
    s_rank: int,
    rank: int,
    iters: int,
    v_scaling: int = 2,
    calculate_loss: bool = True,
    reg: float = 1.0,
    hide_bar: bool = False
) -> StratifiedNMFReturnType:
    """Runs Stratified-NMF on the given data.

    Args:
        A: list of data matrices
        rank: rank to use for W's and H
        iters: iterations to run
        v_scaling: Number of times to update v each iteration. Defaults to 2.
        calculate_loss: Whether to calculate the loss. Defaults to True.

    Returns:
        V: learned V
        X: learned X
        W: learned W
        H: learned H
        loss_array: loss at each iteration.
            Return a zeros array if calculate_loss is False.
    """

    # Constants
    strata = len(A)
    cols = A[0].shape[1]

    # Initialize V, W, H
    V = [np.random.rand(A[i].shape[0], s_rank) / rank**0.5 for i in range(strata)]
    X = [np.random.rand(s_rank, A[i].shape[1]) / rank**0.5 for i in range(strata)]
    W = [np.random.rand(A[i].shape[0], rank) / rank**0.5 for i in range(strata)]
    H = np.random.rand(rank, cols) / rank**0.5

    # Keep track of loss array
    loss_array = np.zeros(iters)

    if isinstance(A[0], csr_array) and calculate_loss:
        cprint(
            "Warning: loss calculation decreases performance when using large, sparse matrices.",
            "yellow",
        )

    # Run NMF
    for i in trange(iters, disable=hide_bar):

        # Calculate loss
        if calculate_loss:
            loss_array[i] = loss(A, V, X, W, H)

        # Update V
        for _ in range(v_scaling):
            V, X = update_V(A, V, X, W, H), update_X(A, V, X, W, H, reg)

            #print("X mean: ", [X[j].mean() for j in range(len(X))])

        # Update W, H
        W, H = update_W(A, V, X, W, H), update_H(A, V, X, W, H, reg)
        #print("X mean end: ", [X[j].mean() for j in range(len(X))])
    assert np.all(H >= 0)
    for s in range(strata):
        assert np.all(V[s] >= 0)
        assert np.all(X[s] >= 0)
        assert np.all(W[s] >= 0)
    
    return V, X, W, H, loss_array


if __name__ == "__main__":
    # This demonstrates that the code works on random data with different strata sizes.
    strata_test = 2
    rows_test = [100, 200]
    cols_test = 100
    srank_test = 2
    rank_test = 10
    iters_test = 1000
    #A_test = [np.random.rand(rows_test[s], cols_test) for s in range(strata_test)]
    
    V = [np.random.rand(rows_test[s], srank_test) for s in range(strata_test)]
    X = [np.random.rand(srank_test, cols_test) for s in range(strata_test)]
    W = [np.random.rand(rows_test[s], rank_test) for s in range(strata_test)]
    H = np.random.rand(rank_test, cols_test)
    
    A_test = [V[s] @ X[s] + W[s] @ H for s in range(strata_test)]
    # Run NMF
    V, X, W, H, loss_array_test = estratified_nmf(A_test, srank_test, rank_test, iters_test)

    # Plot loss
    plt.plot(loss_array_test)
    plt.show()
