import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    exp_name = "mnist_final_0.01"
    train_filename = os.path.join(os.getcwd(), "logs", f"{exp_name}_step.csv")
    val_filename = os.path.join(os.getcwd(), "logs", f"{exp_name}_epoch.csv")
    train_data = pd.read_csv(train_filename)
    val_data = pd.read_csv(val_filename)

    plt.plot(train_data.step, train_data.train_loss)
    plt.title(f"Train loss per step, lr={exp_name.split('_')[-1]}")
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.ylim(0.00, 5.0)
    plt.savefig(f"logs/{exp_name}_step.png")
    plt.clf()

    plt.plot(val_data.step, val_data.val_loss, label="Validation loss")
    plt.plot(val_data.step, val_data.train_loss, label="Train loss")
    plt.title(f"Mean loss per epoch, lr={exp_name.split('_')[-1]}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.savefig(f"logs/{exp_name}_val_epoch.png")
    plt.clf()

    plt.plot(val_data.step, val_data.accuracy)
    plt.title(f"Accuracy on test set, lr={exp_name.split('_')[-1]}")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(f"logs/{exp_name}_acc_epoch.png")


if __name__ == "__main__":
    main()
