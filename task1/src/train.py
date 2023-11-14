import time
import os
import numpy as np


def train_loop(
    exp_name,
    epochs,
    xtrain,
    ytrain,
    xval,
    yval,
    nn,
    loss,
    lr,
    batch_size,
    log_interval=None,
):
    step_filepath = os.path.join(os.getcwd(), "logs", f"{exp_name}_step.csv")
    epoch_filepath = os.path.join(os.getcwd(), "logs", f"{exp_name}_epoch.csv")
    step_log_file = open(step_filepath, "w")
    step_log_file.write("step,train_loss\n")
    epoch_log_file = open(epoch_filepath, "w")
    epoch_log_file.write("step,train_loss,val_loss,accuracy\n")

    for i in range(epochs):
        start_time = time.time()
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
                log_step_data(
                    step_log_file,
                    i * len(xtrain) // batch_size + j // batch_size,
                    batch_loss / batch_size,
                )
                mean_loss += batch_loss
                batch_loss = 0.0

            if log_interval is not None and j % log_interval == 0:
                print(f"Epoch: {i} iter: {j} loss: {loss_val:.4}")
        end_time = time.time()
        train_mean_loss = mean_loss / len(xtrain)
        mean_loss = 0.0
        predictions = []
        for j, (x, y) in enumerate(zip(xval, yval)):
            pred = nn.forward(x)
            predictions.append(pred)
            loss_val, loss_grad = loss(pred, y)
            mean_loss += loss_val
        val_mean_loss = mean_loss / len(xval)
        predictions = np.array(predictions).argmax(axis=1)
        acc = calc_accuracy(predictions, yval)
        log_epoch_data(epoch_log_file, i, train_mean_loss, val_mean_loss, acc)
        print(
            f"Epoch: {i} train loss: {train_mean_loss:.4}, "
            f"val loss: {val_mean_loss:.4}, accuracy {acc}, "
            f"time: {end_time - start_time:.2}s"
        )
    step_log_file.close()
    epoch_log_file.close()


def calc_accuracy(pred, label):
    return np.sum(pred == label) / len(pred)


def log_step_data(file, step, train_loss):
    file.write(f"{step},{train_loss}\n")
    file.flush()


def log_epoch_data(file, step, train_loss, val_loss, acc):
    file.write(f"{step},{train_loss},{val_loss},{acc}\n")
    file.flush()
