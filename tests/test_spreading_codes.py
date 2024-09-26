from decor.spreading_codes import SpreadingCodes, randb
import numpy as np


def test_randb():
    codes = randb(5, 31)
    assert codes.shape == (5, 31)
    assert np.all(np.isclose(codes, 1) | np.isclose(codes, -1))


def test_properties():
    value = np.random.choice((-1.0, 1.0), (5, 31))
    x = SpreadingCodes(value=value)
    assert x.shape == (5, 31)
    for i in range(5):
        for j in range(31):
            assert x[i, j] == value[i, j]


def test_correlation_values():
    x = SpreadingCodes(5, 31)
    ffts = np.fft.fft(x.value, axis=1)
    idx = 0
    for i in range(5):
        for j in range(i, 5):
            true = np.fft.ifft(ffts[i] * np.conj(ffts[j])).real / 31
            if i == j:
                true[0] = 0.0
            assert np.allclose(x.correlation()[idx], true)
            idx += 1


def test_objective():
    x = SpreadingCodes(5, 31, p=3)
    assert np.allclose(x.objective(), (np.abs(x.correlation()) ** 3).sum() ** (1 / 3))


def test_deltas():
    x = SpreadingCodes(5, 31, p=3)

    def _obj(x, p):
        ffts = np.fft.fft(x, axis=1)
        res = 0.0
        for i in range(x.shape[0]):
            for j in range(i, x.shape[0]):
                corr_ij = np.fft.ifft(ffts[i] * np.conj(ffts[j])).real / 31
                if i == j:
                    corr_ij[0] = 0.0
                res += np.sum(np.abs(corr_ij) ** p)
        return res

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_cp = x.value.copy()
            x_cp[i, j] *= -1
            assert np.allclose(x.deltas()[i, j], _obj(x_cp, 3) - _obj(x.value, 3))


def test_best_delta():
    x = SpreadingCodes(5, 31, p=3)
    improvement = x.deltas()
    assert np.allclose(improvement.min(), improvement[x.best_delta()])


def test_copy():
    codes = randb(5, 31)
    x = SpreadingCodes(value=codes, p=3)
    x.deltas()
    x.objective()
    y = x.deepcopy()
    assert np.allclose(
        x._correlation,
        y._correlation,
    )
    assert np.isclose(x._objective, y._objective)
    assert np.allclose(x._delta, y._delta)


def test_flip_bit():
    x = randb(5, 31)
    y = x.copy()
    y[2, 21] *= -1
    _x = SpreadingCodes(value=x, p=3)
    _y = SpreadingCodes(value=y, p=3)

    _x.flip(2, 21)
    assert np.allclose(_x.deltas(), _y.deltas())
    assert np.allclose(_x.objective(), _y.objective())
    assert np.allclose(_x.correlation(), _y.correlation())
    assert np.allclose(_x.value, _y.value)


def test_top_k_delta():
    x = SpreadingCodes(5, 31, p=3)
    tups = x.top_k_delta()
    assert np.allclose(
        np.sort(x.deltas().flatten()),
        [x.deltas()[*tup] for tup in tups],
    )

    tups = x.top_k_delta(k=7)
    assert np.allclose(
        np.sort(x.deltas().flatten())[:7],
        [x.deltas()[*tup] for tup in tups],
    )


def test_update_deltas():
    x = SpreadingCodes(5, 31, p=3)
    x.deltas()
    for _ in range(10):
        x.flip(np.random.choice(5), np.random.choice(31))
    y = x.copy()
    assert np.allclose(x.deltas(), y.deltas())
