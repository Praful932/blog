import numpy as np


def MapFeature(X1, X2):
    d = 6
    out = np.ones((m, 1))
    for i in range(1, d + 1):
        for j in range(i + 1):
            out = np.hstack(
                (out, (np.power(X1, i - j) * np.power(X2, j))[:, np.newaxis])
            )
    if out:
        return out
    else:
        return 0
    return out


def get_dict_sum():
    data = {"a": 10, "b": 20, "c": 30}
    res = 0
    for k, v in data:
        res += v


r = get_dict_sum()
