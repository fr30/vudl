import math
import numpy as np
import random


def softmax(x):
    exps = [math.exp(i) for i in x]
    out = [i / sum(exps) for i in exps]
    return out


def np_softmax(x):
    return np.exp(x) / np.exp(x).sum()


def add(x1, x2):
    if isinstance(x1[0], list) and isinstance(x2[0], list):
        x = []
        for i in range(len(x1)):
            row = []
            for j in range(len(x1[i])):
                row.append(x1[i][j] + x2[i][j])
            x.append(row)
        return x
    elif not isinstance(x1[0], list) and not isinstance(x2[0], list):
        return [e1 + e2 for e1, e2 in zip(x1, x2)]
    else:
        return None


def scalar_mul(a, x):
    if isinstance(x[0], list):
        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] *= a
        return x
    else:
        return [a * i for i in x]


def mul(x1, x2):
    return [e1 * e2 for e1, e2 in zip(x1, x2)]


def matmul(left, right):
    if isinstance(left[0], list) and isinstance(right[0], list):
        if len(left[0]) != len(right):
            raise ValueError("Matrix dimension mismatch")
        res = []
        for i in range(len(left)):
            row = []
            for j in range(len(right[0])):
                s = 0
                for k in range(len(left[0])):
                    s += left[i][k] * right[k][j]
                row.append(s)
            res.append(row)
        return res
    elif not isinstance(left[0], list) and not isinstance(right[0], list):
        out = []
        for i in range(len(left)):
            row = []
            for j in range(len(right)):
                row.append(left[i] * right[j])
            out.append(row)
        return out

    elif not isinstance(right[0], list):
        left, right = right, left
    res = []
    for i in range(len(right[0])):
        s = 0
        for j in range(len(left)):
            s += left[j] * right[j][i]
        res.append(s)
    return res


def zeros(shape):
    return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]


def init_weights(shape):
    return [[random.gauss(0, 1) for _ in range(shape[1])] for _ in range(shape[0])]
