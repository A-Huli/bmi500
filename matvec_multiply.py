"""Matrix-vector multiplication implemented with Python loops."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Sequence

SIZE = 1000
SEED = 42


# create a function to compute the dot product of two vectors using a for loop
# add comments for the selected function
def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product of two vectors of equal length.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The sum of a[i] * b[i] over all indices i.

    Raises:
        ValueError: If a and b have different lengths.
    """
    # the dot product is only defined for vectors of equal length
    if len(a) != len(b):
        raise ValueError(f"vectors must have the same length, got {len(a)} and {len(b)}")

    # zip pairs corresponding elements
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += x * y
    return total


# create a function to compute the matrix-vector product using the dot_product function
# add comments for the selected function
def matrix_vector_product(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    """Return the product A @ x of an m x n matrix and a length-n vector.

    Args:
        matrix: An m x n matrix.
        vector: A vector of length n.

    Returns:
        A list of length m.

    Raises:
        ValueError: If any row's length differs from the vector's length.
    """
    # one dot product per row
    # dot_product validates each row's length
    return [dot_product(row, vector) for row in matrix]


def random_vector(n: int, rng: random.Random) -> list[float]:
    """Return a vector of ``n`` values drawn uniformly from [0, 1)."""
    return [rng.random() for _ in range(n)]


def random_matrix(rows: int, cols: int, rng: random.Random) -> list[list[float]]:
    """Return a ``rows`` x ``cols`` matrix of values drawn uniformly from [0, 1)."""
    return [random_vector(cols, rng) for _ in range(rows)]


# create a main function to test the matrix-vector product function
# using randomly generated data of size 1000x1000
# add comments for the selected function
def main() -> None:
    """Time the product on random 1000x1000 data."""
    # Random data
    random.seed(SEED)
    matrix = [[random.random() for _ in range(SIZE)] for _ in range(SIZE)]
    vector = [random.random() for _ in range(SIZE)]

    # Time of the multiplication.
    start = time.perf_counter()
    result = matrix_vector_product(matrix, vector)
    elapsed = time.perf_counter() - start

    print(f"A {SIZE}x{SIZE} couted in in {elapsed:.4f} s")
    print(f"First 10 values: {[round(x, 4) for x in result[:10]]}")



if __name__ == "__main__":
    main()
