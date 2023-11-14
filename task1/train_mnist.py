from src.numpy_nn import NeuralNetwork
from src.data import load_mnist
from src.loss import np_log_crossentropy
from src.train import train_loop


def normalize_image(image):
    return image / 255.0


log_interval = None
epochs = 5
batch_size = 64
# lr = 1e-3
# lr = 3e-3
# lr = 1e-2
lr = 1e-2
loss = np_log_crossentropy
nn = NeuralNetwork(input_size=784, hidden_size=300, out_size=10, return_logits=True)
(xtrain, ytrain), (xval, yval), num_cls = load_mnist(flatten=True, final=True)
xtrain, xval = normalize_image(xtrain), normalize_image(xval)
exp_name = f"mnist_final_{lr}"

train_loop(
    exp_name, epochs, xtrain, ytrain, xval, yval, nn, loss, lr, batch_size, log_interval
)
