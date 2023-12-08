import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 0.003

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.batch_norm0 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 240)
        self.batch_norm2 = nn.BatchNorm1d(240)
        self.fc2 = nn.Linear(240, 168)
        self.batch_norm3 = nn.BatchNorm1d(168)
        self.fc3 = nn.Linear(168, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm0(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


net = Net().to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


for epoch in range(NUM_EPOCHS):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1]
            # calculate outputs by running images through the network
            outputs = net(images).cpu()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"EPOCH {epoch:2}")
    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:  # print every 2000 mini-batches
        #     print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
        #     running_loss = 0.0
    print(f"Running loss {running_loss:.3f}")
print("Finished Training")
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1]
        # calculate outputs by running images through the network
        outputs = net(images).cpu()
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
