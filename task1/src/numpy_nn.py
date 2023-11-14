import numpy as np


class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=3, out_size=2, return_logits=False):
        self.return_logits = return_logits
        # self.w = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        self.w = np.random.normal(size=(input_size, hidden_size))
        # self.w = np.array(
        #     [
        #         [0.07404875, -0.71184316, -0.12654778],
        #         [0.54491774, 0.75364024, -1.10045576],
        #     ]
        # )
        self.b = np.zeros(hidden_size)
        # self.v = np.array([[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])
        self.v = np.random.normal(size=(hidden_size, out_size))
        # self.v = np.array(
        #     [
        #         [-0.17663458, 0.83504845],
        #         [-1.15204888, -1.25854565],
        #         [-0.02037359, -1.21206247],
        #     ]
        # )
        self.c = np.zeros(out_size)
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
            grad = np.dot(self.softmax.grad(), y_grad)
        local_grad, local_v_grad, local_c_grad = self.l2.grad()
        grad = np.expand_dims(grad, axis=0)
        local_v_grad = np.expand_dims(local_v_grad, axis=1)
        # dl/dv 3x2
        self.v_grad = self.v_grad + np.dot(local_v_grad, grad)
        # dl/dc 2x1
        self.c_grad = self.c_grad + np.multiply(local_c_grad, grad.flatten())
        # dl/dh 3x1
        grad = np.dot(grad, local_grad)
        local_grad, local_w_grad, local_b_grad = self.l1.grad()
        # dl/dw 2x3
        local_w_grad = np.expand_dims(local_w_grad, axis=1)
        self.w_grad = self.w_grad + np.dot(local_w_grad, grad)
        # dl/db 3x1
        self.b_grad = self.b_grad + np.multiply(local_b_grad, grad.flatten())

    def reset_grad(self):
        self.w_grad = np.zeros(self.w.shape)
        self.b_grad = np.zeros(self.b.shape)
        self.v_grad = np.zeros(self.v.shape)
        self.c_grad = np.zeros(self.c.shape)

    def apply_gradient(self, lr=0.001):
        self.w = self.w - lr * self.w_grad
        self.b = self.b - lr * self.b_grad
        self.v = self.v - lr * self.v_grad
        self.c = self.c - lr * self.c_grad
        self.l1.update_params(self.w, self.b)
        self.l2.update_params(self.v, self.c)


class Linear:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def __call__(self, x):
        self.input_x = x
        x = np.dot(x, self.W)
        x = x + self.b
        return x

    def grad(self):
        return np.transpose(self.W), self.input_x, np.ones(self.b.shape)

    def update_params(self, W, b):
        self.W = W
        self.b = b


class Softmax:
    def __init__(self):
        self.out = None

    def __call__(self, x):
        out = np.exp(x) / np.exp(x).sum()
        self.out = out
        return out

    def grad(self):
        i = np.arange(len(self.out))
        j = np.arange(len(self.out))
        grid_i, grid_j = np.meshgrid(i, j)
        jacobian = np.where(
            grid_i == grid_j,
            self.out[i] * (1 - self.out[i]),
            -self.out[i] * self.out[j],
        )
        return jacobian


class Sigmoid:
    def __call__(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def grad(self):
        return self.out * (1 - self.out)
