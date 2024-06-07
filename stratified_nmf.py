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
    V: np.ndarray,
    W: list[np.ndarray],
    H: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Updates V using the multiplicative update rule."""
    strata = len(A)
    out = np.zeros(V.shape)

    for s in range(strata):
        rows = A[s].shape[0]
        den = V[s] * rows + H.T @ np.sum(W[s], axis=0) + tol
        if isinstance(A[s], csr_array):
            out[s] = V[s] * A[s].sum(axis=0) / den
        else:
            out[s] = V[s] * np.sum(A[s], axis=0) / den

    return out


def update_W(
    A: list[np.ndarray],
    V: np.ndarray,
    W: list[np.ndarray],
    H: np.ndarray,
    tol: float = 1e-9,
) -> list[np.ndarray]:
    """Updates W using the multiplicative update rule."""
    strata = len(A)
    out = [np.zeros(W[s].shape) for s in range(strata)]

    for s in range(strata):
        rows = A[s].shape[0]
        den = np.dot(W[s], np.dot(H, H.T)) + np.outer(np.ones(rows), V[s] @ H.T) + tol
        if isinstance(A[s], csr_array):
            out[s] = W[s] * A[s].dot(H.T) / den
        else:
            out[s] = W[s] * np.dot(A[s], H.T) / den

    return out


def update_H(
    A: list[np.ndarray],
    V: np.ndarray,
    W: list[np.ndarray],
    H: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Updates H using the multiplicative update rule."""
    strata = len(A)
    out = np.zeros(H.shape)

    num = 0
    den = 0

    for s in range(strata):
        if isinstance(A[s], csr_array):
            num += ((A[s].T).dot(W[s])).T
        else:
            num += np.dot(W[s].T, A[s])
        den += np.outer(np.sum(W[s], axis=0), V[s]) + np.dot(np.dot(W[s].T, W[s]), H)
    out = H * num / (den + tol)

    return out


def loss(
    A: list[np.ndarray],
    V: np.ndarray,
    W: list[np.ndarray],
    H: np.ndarray,
) -> float:
    """Calculates the loss sqrt(sum_s ||A(s) - 1 v(s)^T - W(s) H||_F^2 )"""
    strata = len(A)
    out = 0.0
    for s in range(strata):
        rows = A[s].shape[0]
        out += (
            np.linalg.norm(A[s] - np.dot(W[s], H) - np.outer(np.ones(rows), V[s])) ** 2
        )
    return out**0.5


def stratified_nmf(
    A: list[np.ndarray],
    rank: int,
    iters: int,
    v_scaling: int = 2,
    calculate_loss: bool = True,
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
        W: learned W
        H: learned H
        loss_array: loss at each iteration.
            Return a zeros array if calculate_loss is False.
    """

    # Constants
    strata = len(A)
    cols = A[0].shape[1]

    # Initialize V, W, H
    V = np.random.rand(strata, cols)
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
            loss_array[i] = loss(A, V, W, H)

        # Update V
        for _ in range(v_scaling):
            V = update_V(A, V, W, H)

        # Update W, H
        W, H = update_W(A, V, W, H), update_H(A, V, W, H)

    assert np.all(V >= 0) and np.all(H >= 0)
    for s in range(strata):
        assert np.all(W[s] >= 0)

    return V, W, H, loss_array


if __name__ == "__main__":
    # This demonstrates that the code works on random data with different strata sizes.
    strata_test = 2
    rows_test = [821, 908]
    cols_test = 10447
    rank_test = 10
    iters_test = 100
    A_test = [np.random.rand(rows_test[s], cols_test) for s in range(strata_test)]

    # Run NMF
    _, _, _, loss_array_test = stratified_nmf(A_test, rank_test, iters_test)

    # Plot loss
    plt.plot(loss_array_test)
    plt.show()
