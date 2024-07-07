from torchvision import transforms
import glob
from PIL import Image
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model_v2 import MobileNetV2

import os

# 定义数据转换
test_data_transforms = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 获取所有图像文件列表
all_images = sorted(glob.glob('TestSets/*.jpg'))  # 修改为你的图像文件目录

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
custom_dataset = CustomDataset(images, labels, transform=test_data_transforms)



# 加载保存的模型
saved_model_path = 'mob_best_model_acc_3_89.82.pth'  # 替换为你保存的模型路径
loaded_model = torch.load(saved_model_path)
# 创建一个新的模型对象（例如，如果使用的是预训练模型）
model = MobileNetV2(num_classes=10) # 以 ResNet-18 为例
# 从 OrderedDict 中提取模型的状态字典（state_dict）
model.load_state_dict(loaded_model)

# 设置计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

# 创建 DataLoader 加载测试集数据
batch_size = 64  # 适当调整 batch size
test_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

# 评估模型
model.eval()
true_labels = []
predicted_labels = []

# 遍历测试集并进行预测
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# 计算评价指标
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

