import math
import numpy as np
from src.utils import mul, add, softmax, np_softmax


def crossentropy(x, y):
    loss_val = -math.log(x[y])
    onehot = [0] * len(x)
    onehot[y] = 1
    grad_elemwise = [-1 / elem for elem in x]
    loss_grad = mul(onehot, grad_elemwise)
    return loss_val, loss_grad


def log_crossentropy(x, y):
    probs = softmax(x)
    loss_val = -math.log(probs[y])
    neg_onehot = [0] * len(x)
    neg_onehot[y] = -1
    loss_grad = add(probs, neg_onehot)
    return loss_val, loss_grad


def np_crossentropy(x, y):
    loss_val = -np.log(x[y])
    onehot = np.zeros(len(x))
    onehot[y] = 1
    loss_grad = onehot * (-1 / x)
    return loss_val, loss_grad


def np_log_crossentropy(x, y):
    probs = np_softmax(x)
    loss_val = -np.log(probs[y])
    onehot = np.zeros(len(x))
    onehot[y] = 1
    loss_grad = probs - onehot
    return loss_val, loss_grad
