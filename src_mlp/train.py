import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def fix_seed(seed=0):
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


fix_seed()

train_dataset = torchvision.datasets.MNIST(
    root="data/input/train",
    train=True,
    transform=transforms.ToTensor(),
    download=False,
)

test_dataset = torchvision.datasets.MNIST(
    root="data/input/train",
    train=False,
    transform=transforms.ToTensor(),
    download=False,
)

batch_size = 256

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

num_classes = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 最適化手法を設定
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train_fn(model, train_loader, criterion, optimizer, device="cpu"):

    # 1epoch training
    train_loss = 0.0
    num_train = 0

    # model 学習モードに設定
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        # batch数を累積
        num_train += len(labels)

        # viewで1次元配列に変更
        # toでgpuに転送
        images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
        # 勾配をリセット
        optimizer.zero_grad()
        # 推論
        outputs = model(images)
        # lossを計算
        loss = criterion(outputs, labels)
        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()
        # lossを累積
        train_loss += loss.item()

    train_loss = train_loss / num_train

    return train_loss


def valid_fn(model, train_loader, criterion, optimizer, device="cpu"):

    # 評価用のコード
    valid_loss = 0.0
    num_valid = 0

    # model 評価モードに設定
    model.eval()

    # 評価の際に勾配を計算しないようにする
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


def main():
    def run(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device="cpu",
        num_epochs=10,
    ):

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

    os.makedirs("data/output", exist_ok=True)
    model_path = "data/output/model.pkl"
    pickle.dump(model, open(model_path, "wb"))


if __name__ == "__main__":
    main()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
