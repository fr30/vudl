from src.numpy_nn import NeuralNetwork
from src.plain_nn import PlainNeuralNetwork
from src.data import load_synth
from src.loss import (
    np_log_crossentropy,
    np_crossentropy,
    crossentropy,
    log_crossentropy,
)
from src.train import train_loop


log_interval = None
exp_name = "train_np"
epochs = 30
batch_size = 16
lr = 1e-3
loss = np_log_crossentropy
nn = NeuralNetwork(return_logits=True)
(xtrain, ytrain), (xval, yval), num_cls = load_synth()

train_loop(
    exp_name, epochs, xtrain, ytrain, xval, yval, nn, loss, lr, batch_size, log_interval
)
