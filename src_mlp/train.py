import os
import sys
from typing import Any
import cloudpickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# fixed random seed
def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

fix_seed()


# load dataset
train_dataset = torchvision.datasets.MNIST(
    root="data/input",
    train=True,
    transform=transforms.ToTensor(),
    download=False,
)

test_dataset = torchvision.datasets.MNIST(
    root="data/input",
    train=False,
    transform=transforms.ToTensor(),
    download=False,
)


# converts to batch size
batch_size = 256

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


# define network
num_classes = 10    #number of outputs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    # define forward propagation
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# transfers network to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)


# setting up a loss function
criterion = nn.CrossEntropyLoss()


# use SGD as optimization algorithm
optimizer = optim.SGD(model.parameters(), lr=0.01)


# epoch learning
def train_fn(model: Net, train_loader, criterion: nn.CrossEntropyLoss, optimizer: optim.SGD, device="cpu") -> float:

    # 1epoch training
    train_loss = 0.0
    num_train = 0

    # set to model learning mode
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        # accumulate batch counts
        num_train += len(labels)

        # change to 1D array with view
        # transfer to gpu with to
        images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
        # reset gradient
        optimizer.zero_grad()
        # inference
        outputs = model(images)
        # calculate loss
        loss = criterion(outputs, labels)
        # error back propagation
        loss.backward()
        # update parameters
        optimizer.step()
        # accumulate loss
        train_loss += loss.item()

    train_loss = train_loss / num_train

    return train_loss


# inference
def valid_fn(model: Net, train_loader, criterion: nn.CrossEntropyLoss, optimizer: optim.SGD, device="cpu") -> float:

    # code for evaluation
    valid_loss = 0.0
    num_valid = 0

    # model set to evaluation mode
    model.eval()

    # do not calculate gradient during evaluation
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            num_valid += len(labels)
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

        valid_loss = valid_loss / num_valid

    return valid_loss


print("learning...")


# model Learning
def main():
    def run(model: Net, train_loader, test_loader, criterion: nn.CrossEntropyLoss, optimizer: optim.SGD, device="cpu", num_epochs=10) -> list:

        train_loss_list = []
        valid_loss_list = []

        for epoch in range(num_epochs):

            _train_loss = train_fn(
                model, train_loader, criterion, optimizer, device=device
            )
            _valid_loss = valid_fn(
                model, train_loader, criterion, optimizer, device=device
            )

            print(
                f"Epoch [{epoch+1}], train_Loss : {_train_loss:.5f}, val_Loss : {_valid_loss:.5f}"
            )

            train_loss_list.append(_train_loss)
            valid_loss_list.append(_valid_loss)

        return train_loss_list, valid_loss_list

    # save the model
    os.makedirs("data/output", exist_ok=True)
    model_path = "data/output/model.pkl"
    with open(model_path, mode="wb") as out:
        cloudpickle.dump(model, out)


if __name__ == "__main__":
    main()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
