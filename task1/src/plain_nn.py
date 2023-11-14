import math

from src.utils import (
    add,
    init_weights,
    matmul,
    mul,
    scalar_mul,
    zeros,
)


class PlainNeuralNetwork:
    def __init__(self, return_logits=False):
        self.return_logits = return_logits
        # self.w = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
        # self.w = [
        #     [0.07404875, -0.71184316, -0.12654778],
        #     [0.54491774, 0.75364024, -1.10045576],
        # ]
        self.w = init_weights([2, 3])
        self.b = [0.0, 0.0, 0.0]
        # self.v = [[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]
        # self.v = [
        #     [-0.17663458, 0.83504845],
        #     [-1.15204888, -1.25854565],
        #     [-0.02037359, -1.21206247],
        # ]
        self.v = init_weights([3, 2])
        self.c = [0.0, 0.0]
        self.l1 = Linear(self.w, self.b)
        self.l2 = Linear(self.v, self.c)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()
        self.reset_grad()

    def forward(self, x):
        x = self.l1(x)
        x = self.sigmoid(x)
        x = self.l2(x)
        if not self.return_logits:
            x = self.softmax(x)
        return x

    def backward(self, y_grad):
        # dl/do 2x1
        if self.return_logits:
            grad = y_grad
        else:
            grad = matmul(self.softmax.grad(), y_grad)
        local_grad, local_v_grad, local_c_grad = self.l2.grad()
        # dl/dv 2x3
        self.v_grad = add(self.v_grad, matmul(local_v_grad, grad))
        # dl/dc 2x1
        self.c_grad = add(self.c_grad, mul(grad, local_c_grad))
        # dl/dh 3x1
        grad = matmul(grad, local_grad)
        local_grad, local_w_grad, local_b_grad = self.l1.grad()
        # dl/dw 2x3
        self.w_grad = add(self.w_grad, matmul(local_w_grad, grad))
        # dl/db 3x1
        self.b_grad = add(self.b_grad, mul(local_b_grad, grad))

    def reset_grad(self):
        self.w_grad = zeros([2, 3])
        self.b_grad = [0.0, 0.0, 0.0]
        self.v_grad = zeros([3, 2])
        self.c_grad = [0.0, 0.0]

    def apply_gradient(self, lr=0.001):
        self.w = add(self.w, scalar_mul(-lr, self.w_grad))
        self.b = add(self.b, scalar_mul(-lr, self.b_grad))
        self.v = add(self.v, scalar_mul(-lr, self.v_grad))
        self.c = add(self.c, scalar_mul(-lr, self.c_grad))
        self.l1.update_params(self.w, self.b)
        self.l2.update_params(self.v, self.c)


class Linear:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def __call__(self, x):
        self.input_x = x
        x = matmul(self.W, x)
        x = add(x, self.b)
        return x

    def grad(self):
        b_grad = [1] * len(self.b)
        W_t = []
        for i in range(len(self.W[0])):
            row = []
            for j in range(len(self.W)):
                row.append(self.W[j][i])
            W_t.append(row)
        return W_t, self.input_x, b_grad

    def update_params(self, W, b):
        self.W = W
        self.b = b


class Softmax:
    def __init__(self):
        self.out = None

    def __call__(self, x):
        exps = [math.exp(i) for i in x]
        out = [i / sum(exps) for i in exps]
        self.out = out
        return out

    def grad(self):
        jacobian = []
        for i in range(len(self.out)):
            row = []
            for j in range(len(self.out)):
                if i == j:
                    row.append(self.out[i] * (1 - self.out[i]))
                else:
                    row.append(-self.out[i] * self.out[j])
            jacobian.append(row)
        return jacobian


class Sigmoid:
    def __call__(self, x):
        out = [1.0 / (1.0 + math.exp(-i)) for i in x]
        self.out = out
        return out

    def grad(self):
        ones = [1.0 for _ in self.out]
        negout = [-i for i in self.out]
        return mul(self.out, add(ones, negout))
