"""Basic dense matrix multiplication in pure Python (standard library only).

Storage: a matrix is a list of n rows, each row a list of n Python floats
(IEEE-754 double). Elements are boxed float objects, so each element costs a
pointer in the row list plus the float object itself.
"""

import sys
from array import array


def zeros(n):
    """Return an n x n matrix of 0.0 as a list of row lists."""
    return [[0.0] * n for _ in range(n)]


def multiply(a, b, c, n):
    """C = A * B with the basic i-j-k triple loop and a scalar accumulator.

    Performs n^3 multiplications and n^3 additions. Every C[i][j] is assigned,
    so c does not need to be cleared between calls.
    """
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            c[i][j] = s


def checksum(c):
    """Sum of all entries in row-major order, used to consume the result."""
    s = 0.0
    for row in c:
        for x in row:
            s += x
    return s


def load_matrix(path, n):
    """Read a raw little-endian float64 row-major n x n file into row lists.

    Rows are read one at a time so that no full-size temporary buffer exists
    next to the list-of-lists representation.
    """
    expected = 8 * n * n
    with open(path, "rb") as f:
        size = f.seek(0, 2)
        if size != expected:
            raise ValueError(f"{path}: {size} bytes, expected {expected} for n={n}")
        f.seek(0)
        rows = []
        for _ in range(n):
            row = array("d")
            row.fromfile(f, n)
            if sys.byteorder != "little":
                row.byteswap()
            rows.append(row.tolist())
    return rows


def write_matrix(path, c):
    """Write row lists as raw little-endian float64, row-major."""
    with open(path, "wb") as f:
        for r in c:
            row = array("d", r)
            if sys.byteorder != "little":
                row.byteswap()
            row.tofile(f)
