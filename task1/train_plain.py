from src.plain_nn import PlainNeuralNetwork
from src.data import load_synth
from src.loss import log_crossentropy, crossentropy

# nn = PlainNeuralNetwork(return_logits=False)
# x = [1, -1]
# y = 0
# loss = crossentropy
# pred = nn.forward(x)
# loss_val, loss_grad = loss(pred, y)
# nn.backward(loss_grad)

log_interval = None
exp_name = "train_plain"

epochs = 10
batch_size = 16
lr = 1e-3
loss = log_crossentropy
nn = PlainNeuralNetwork(return_logits=True)

(xtrain, ytrain), (xval, yval), num_cls = load_synth()
for i in range(epochs):
    mean_loss = 0.0
    batch_loss = 0.0
    for j, (x, y) in enumerate(zip(xtrain, ytrain)):
        pred = nn.forward(x)
        loss_val, loss_grad = loss(pred, y)
        nn.backward(loss_grad)
        batch_loss += loss_val

        if j % batch_size == 0:
            nn.apply_gradient(lr)
            nn.reset_grad()
            mean_loss += batch_loss
            batch_loss = 0.0

    train_mean_loss = mean_loss / len(xtrain)
    mean_loss = 0.0
    predictions = []
    for j, (x, y) in enumerate(zip(xval, yval)):
        pred = nn.forward(x)
        predictions.append(pred)
        loss_val, loss_grad = loss(pred, y)
        mean_loss += loss_val
    val_mean_loss = mean_loss / len(xval)
    print(
        f"Epoch: {i} train loss: {train_mean_loss:.4}, " f"val loss: {val_mean_loss:.4}"
    )
