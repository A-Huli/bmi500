"""Tests for dot_product and matrix_vector_product in matvec_multiply.py.

Run with:  pytest test_matvec_multiply.py
"""

import random

import pytest

from matvec_multiply import dot_product, matrix_vector_product


# generate tests for matrix-vector-product function in matvec_multiply.py


def test_dot_product_integers():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32.0


def test_dot_product_floats():
    assert dot_product([0.5, 1.5], [2.0, 4.0]) == pytest.approx(7.0)


def test_dot_product_negative_values():
    assert dot_product([-1, 2, -3], [4, -5, 6]) == -32.0


def test_dot_product_orthogonal_vectors_is_zero():
    assert dot_product([1, 0], [0, 1]) == 0.0


def test_dot_product_single_element():
    assert dot_product([3], [7]) == 21.0


def test_dot_product_empty_vectors_is_zero():
    assert dot_product([], []) == 0.0


def test_dot_product_returns_float():
    assert isinstance(dot_product([1, 2], [3, 4]), float)


def test_dot_product_is_commutative():
    a = [1.2, -3.4, 5.6]
    b = [7.8, 9.0, -1.1]
    assert dot_product(a, b) == pytest.approx(dot_product(b, a))


def test_dot_product_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        dot_product([1, 2, 3], [1, 2])

# generate tests for dot-product function in matvec_multiply.py

def test_matvec_square_matrix():
    matrix = [[1, 2], [3, 4]]
    vector = [5, 6]
    assert matrix_vector_product(matrix, vector) == [17.0, 39.0]


def test_matvec_non_square_matrix():
    # 2x3 matrix times a length-3 vector gives a length-2 result
    matrix = [[1, 2, 3], [4, 5, 6]]
    vector = [1, 0, -1]
    assert matrix_vector_product(matrix, vector) == [-2.0, -2.0]

def test_matvec_zero_matrix_returns_zeros():
    zeros = [[0, 0], [0, 0], [0, 0]]
    assert matrix_vector_product(zeros, [9, 9]) == [0.0, 0.0, 0.0]


def test_matvec_zero_vector_returns_zeros():
    assert matrix_vector_product([[1, 2], [3, 4]], [0, 0]) == [0.0, 0.0]


def test_matvec_empty_matrix_returns_empty_list():
    assert matrix_vector_product([], [1, 2, 3]) == []

def test_matvec_row_length_mismatch_raises():
    with pytest.raises(ValueError):
        matrix_vector_product([[1, 2], [3, 4]], [1, 2, 3])
