import os
import shutil
import random


data_dir = "./flowers"  # 五分类数据集路径
train_dir = "./split/train"  # 训练集路径
val_dir = "./split/val"  # 验证集路径
test_dir = "./split/test"  # 测试集路径

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
categories = os.listdir(data_dir)

for category in categories:
    category_dir = os.path.join(data_dir, category)
    files = os.listdir(category_dir)
    random.shuffle(files)

    train_files = files[:int(len(files) * 0.7)]
    val_files = files[int(len(files) * 0.7):int(len(files) * 0.85)]
    test_files = files[int(len(files) * 0.85):]

    for file in train_files:
        src_path = os.path.join(category_dir, file)
        dst_path = os.path.join(train_dir, category, file)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

    for file in val_files:
        src_path = os.path.join(category_dir, file)
        dst_path = os.path.join(val_dir, category, file)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

    for file in test_files:
        src_path = os.path.join(category_dir, file)
        dst_path = os.path.join(test_dir, category, file)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
