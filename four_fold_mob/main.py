
import glob
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from torch.optim import lr_scheduler

from model_v2 import MobileNetV2



# 定义数据转换
train_data_transforms =  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 随机将图片水平翻转
    transforms.RandomRotation(15), # 随机旋转图片
    transforms.ToTensor(), # 将图片转成 Tensor，并把数值normalization到[0,1]
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

# 定义数据转换
test_data_transforms = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 获取所有图像文件列表
all_images = sorted(glob.glob('datasets/*.jpg'))  # 修改为你的图像文件目录

# 处理图像和标签
images = []
labels = []
# 统计各个类别的数据量
label_counter = Counter()
for img_path in all_images:
    filename = os.path.basename(img_path)
    label = filename[0]  # 获取文件名的第一个字符作为类别标签
    labels.append(int(label))
    label_counter[label] += 1
    # 读取图像并进行转换
    img = Image.open(img_path).convert('RGB')
    images.append(img)


# 打印各个类别的数据量
for label, count in label_counter.items():
    print(f"Class {label}: {count} images")
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 如果图像不是 PIL 图像，则转换为 PIL 图像
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)  # 将 Tensor 转换为 PIL 图像

        if self.transform:
            image = self.transform(image)

        return image, label

# 创建自定义数据集实例
custom_dataset = CustomDataset(images, labels, transform=train_data_transforms)

# 创建自定义数据集实例
custom_dataset2 = CustomDataset(images, labels, transform=test_data_transforms)

# 定义四折交叉验证
num_splits = 4

skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)  # 创建StratifiedKFold对象

fold = 0
for train_idx, val_idx in skf.split(images, labels):  # 这里用 images 作为划分数据的基础

    train_subset = torch.utils.data.Subset(custom_dataset, train_idx)
    val_subset = torch.utils.data.Subset(custom_dataset2, val_idx)

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=True)
    # create model
    model = MobileNetV2(num_classes=10)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "mobilenet_v2-b0353104.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in model.features.parameters():
        param.requires_grad = True
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    '''
    num_classes = 10

    # 加载 Resmodel-18 模型，pretrained=False 表示不加载预训练权重
     = models.resnet18(pretrained=False)

    # 获取 ResNet-18 的最后一个全连接层之前的部分（去掉分类器）
    num_ftrs = model.fc.in_features

    # 定义新的分类器
    classifier = nn.Sequential(
        nn.Linear(num_ftrs, num_classes) # 输出层，假设有10个类别
    )

    # 替换 ResNet-18 模型的分类器部分
    model.fc = classifier'''

    '''

    num_classes = 10

    # 加载 MobileNet 模型，pretrained=False 表示不加载预训练权重
    model = models.mobilenet_v2(pretrained=False)

    # 获取 MobileNet 的最后一个全连接层之前的部分（去掉分类器）
    num_ftrs = model.classifier[1].in_features
    classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1024),  # 添加一个线性层
        nn.ReLU(inplace=True),  # ReLU 激活函数
        nn.Linear(1024, num_classes)  # 输出层，假设有10个类别
    )

    # 替换 MobileNet 模型的最后一个全连接层
    model.classifier = classifier'''



    '''

    num_classes = 10
    # Define the ResNet-50 model
    model = models.resnet50(pretrained=False)  # Initialize ResNet-50 without pre-trained weights
    num_ftrs = model.fc.in_features

    # 修改全连接层的输出大小为您数据集中的类别数量
    num_classes = 10  # 假设有10个类别

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),  # 添加全连接层
        nn.ReLU(inplace=True),  # ReLU 激活函数
        nn.Linear(1024, num_classes)
    ) 

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 设置学习率调整函数
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) '''


    # 将模型移至 GPU

    # Training loop
    num_epochs = 30
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0  # 用于保存最佳验证准确率对应的模型

    best_model_path = 'best_model_acc_0.00.pth'  # 初始最佳模型路径

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # 训练过程中使用进度条
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for inputs, labels in tepoch:
                inputs = inputs.to(device)

                # 检查 labels 是否是一个元组，若是，则将其中的张量分别移动到设备
                if isinstance(labels, tuple):
                    labels = tuple(label.to(device) for label in labels)
                else:
                    labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                tepoch.set_postfix(train_loss=running_train_loss / (tepoch.n + 1),
                                   train_acc=100. * correct_train / total_train)

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # 验证过程
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)

                # 检查 labels 是否是一个元组，若是，则将其中的张量分别移动到设备
                if isinstance(labels, tuple):
                    labels = tuple(label.to(device) for label in labels)
                else:
                    labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = 100. * correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        # 保存最佳模型
        if epoch_val_acc > best_val_acc:

            best_val_acc = epoch_val_acc

            best_model_path = f'mob_best_model_acc_{fold}_{best_val_acc:.2f}.pth'  # 更新最佳模型路径

            torch.save(model.state_dict(), best_model_path)  # 保存最佳模型参数到 best_model.pth 文件


    # 绘制训练和验证集的 loss 和 accuracy 曲线
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('mob_train_process'+'%i'%fold+'.jpg')
    fold += 1








