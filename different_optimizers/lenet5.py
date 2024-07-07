import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torchvision


# 定义 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)


# 定义数据预处理操作
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 定义数据预处理操作
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10('data/', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10('data/', train=False, transform=transform2)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


def train(model, optimizer, device):
    model.train()
    train_loss = []
    train_acc = []

    valid_losses = []
    valid_accs = []

    for epoch in range(200):
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(total=len(train_loader), ncols=80) as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()


                pbar.update(1)
                pbar.set_description(f'Epoch {epoch+1}')
                pbar.set_postfix({'Loss': running_loss / (batch_idx+1), 'Accuracy': 100. * correct / total})

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / total)

        # 在验证集上进行评估
        valid_running_loss = 0.0
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for valid_inputs, valid_labels in test_loader:
                # 省略将数据移到 GPU 上的代码
                valid_inputs = valid_inputs.to(device)
                valid_labels = valid_labels.to(device)

                valid_outputs = model(valid_inputs)
                valid_loss = F.cross_entropy(valid_outputs, valid_labels)
                valid_running_loss += valid_loss.item()
                _, valid_predicted = torch.max(valid_outputs.data, 1)
                valid_total += valid_labels.size(0)
                valid_correct += (valid_predicted == valid_labels).sum().item()

        valid_losses.append(valid_running_loss / len(test_loader))
        valid_accs.append(valid_correct / valid_total)


    return train_loss, train_acc, valid_losses, valid_accs


# 使用 SGD 优化器进行训练
optimizer_sgd = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
train_loss_sgd, train_acc_sgd, valid_loss_sgd, valid_acc_sgd = train(model, optimizer_sgd, device)
print('使用 SGD 优化器进行训练完成')

# 使用 Adam 优化器进行训练
optimizer_adam = optim.Adam(model.parameters(), lr=0.0001)
train_loss_adam, train_acc_adam, valid_loss_adam, valid_acc_adam = train(model, optimizer_adam, device)
print('使用 Adam 优化器进行训练完成')

# 使用 RMSprop 优化器进行训练
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.0001)
train_loss_rmsprop, train_acc_rmsprop, valid_loss_rmsprop, valid_acc_rmsprop = train(model, optimizer_rmsprop, device)
print('使用 RMSprop 优化器进行训练完成')

# 绘制损失值的图表
plt.subplot(2, 2, 1)
plt.plot(range(len(train_loss_sgd)), train_loss_sgd, label='SGD')
plt.plot(range(len(train_loss_adam)), train_loss_adam, label='Adam')
plt.plot(range(len(train_loss_rmsprop)), train_loss_rmsprop, label='RMSprop')
plt.ylabel('Train Loss')
plt.legend()

# 绘制准确率的图表
plt.subplot(2, 2, 2)
plt.plot(range(len(train_acc_sgd)), train_acc_sgd, label='SGD')
plt.plot(range(len(train_acc_adam)), train_acc_adam, label='Adam')
plt.plot(range(len(train_acc_rmsprop)), train_acc_rmsprop, label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.legend()


# 绘制损失值的图表
plt.subplot(2, 2, 3)
plt.plot(range(len(valid_loss_sgd)), valid_loss_sgd, label='SGD')
plt.plot(range(len(valid_loss_adam)), valid_loss_adam, label='Adam')
plt.plot(range(len(train_loss_rmsprop)), valid_loss_rmsprop, label='RMSprop')
plt.ylabel('Valid Loss')
plt.legend()

# 绘制准确率的图表
plt.subplot(2, 2, 4)
plt.plot(range(len(valid_acc_sgd)), valid_acc_sgd, label='SGD')
plt.plot(range(len(valid_loss_adam)), valid_acc_adam, label='Adam')
plt.plot(range(len(valid_acc_rmsprop)), valid_acc_rmsprop, label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Valid Accuracy')
plt.legend()

plt.tight_layout()  # 调整子图位置，避免标签重叠
plt.savefig('0.001_loss_acc.jpg')

