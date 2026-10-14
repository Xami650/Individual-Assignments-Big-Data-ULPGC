import csv
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from matmul import checksum, load_matrix, multiply, write_matrix, zeros

MAIN = Path(__file__).resolve().parent.parent / "main.py"
U = 2.0 ** -53


def product(a, b):
    n = len(a)
    c = zeros(n)
    multiply(a, b, c, n)
    return c


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def random_matrix(n, seed):
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]


def test_hand_computed_2x2():
    assert product([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]) == [[19.0, 22.0], [43.0, 50.0]]


def test_hand_computed_3x3_not_commutative():
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    b = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
    assert product(a, b) == [[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]]
    assert product(b, a) == [[90.0, 114.0, 138.0], [54.0, 69.0, 84.0], [18.0, 24.0, 30.0]]


def test_one_by_one():
    assert product([[3.0]], [[-2.5]]) == [[-7.5]]


@pytest.mark.parametrize("n", [1, 4, 7])
def test_identity_is_exact(n):
    a = random_matrix(n, seed=n)
    assert product(a, identity(n)) == a
    assert product(identity(n), a) == a


@pytest.mark.parametrize("n", [1, 5])
def test_zero(n):
    a = random_matrix(n, seed=10 + n)
    assert product(a, zeros(n)) == zeros(n)
    assert product(zeros(n), a) == zeros(n)


def test_output_buffer_is_overwritten():
    a, b = [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]
    c = [[99.0, 99.0], [99.0, 99.0]]
    multiply(a, b, c, 2)
    multiply(a, b, c, 2)
    assert c == [[19.0, 22.0], [43.0, 50.0]]


@pytest.mark.parametrize("n", [10, 33])
def test_matches_numpy_within_tolerance(n):
    a, b = random_matrix(n, seed=100 + n), random_matrix(n, seed=200 + n)
    c = np.array(product(a, b))
    ref = np.array(a) @ np.array(b)
    gamma = n * U / (1 - n * U)
    bound = 2 * gamma * (np.abs(np.array(a)) @ np.abs(np.array(b)))
    assert np.all(np.abs(c - ref) <= bound)


def test_checksum_sums_all_entries():
    assert checksum([[1.0, 2.0], [3.0, 4.5]]) == 10.5


def test_write_load_roundtrip(tmp_path):
    a = random_matrix(6, seed=7)
    path = tmp_path / "m.bin"
    write_matrix(path, a)
    assert path.stat().st_size == 8 * 36
    assert load_matrix(path, 6) == a
    assert np.array_equal(np.fromfile(path, dtype="<f8").reshape(6, 6), np.array(a))


def test_load_rejects_wrong_size(tmp_path):
    path = tmp_path / "m.bin"
    write_matrix(path, zeros(3))
    with pytest.raises(ValueError):
        load_matrix(path, 4)


def test_cli_output_contract(tmp_path):
    a, b = [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]
    write_matrix(tmp_path / "a.bin", a)
    write_matrix(tmp_path / "b.bin", b)
    out = tmp_path / "c.bin"
    result = subprocess.run(
        [sys.executable, str(MAIN), "--n", "2", "--a", str(tmp_path / "a.bin"), "--b", str(tmp_path / "b.bin"),
         "--warmup", "2", "--reps", "3", "--min-sample-seconds", "0.001", "--out", str(out)],
        capture_output=True, text=True, check=True)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    assert [r["kind"] for r in rows] == ["warmup"] * 2 + ["measure"] * 3
    assert all(r["batch"] == "1" for r in rows[:2])
    assert all(int(r["batch"]) >= 1 and int(r["elapsed_ns"]) > 0 for r in rows)
    assert all(float(r["checksum"]) == 134.0 for r in rows)
    assert load_matrix(out, 2) == [[19.0, 22.0], [43.0, 50.0]]
