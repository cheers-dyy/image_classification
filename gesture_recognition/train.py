import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# 定义数据路径和参数
train_dir = "/root/split/train/"
val_dir = "/root/split/val/"
test_dir = "/root/split/test/"
img_height, img_width = 224, 224  # 图像尺寸
batch_size = 32  # 批量大小
num_classes = 10  # 类别数

# 创建AlexNet模型
model = Sequential([
    Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])



# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 加载训练集、验证集和测试集数据
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

# 创建 ModelCheckpoint 回调函数
checkpoint = ModelCheckpoint(
    'path/to/save/best/model.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# 训练模型时添加回调函数
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    validation_data=val_data,
    validation_steps=val_data.samples // batch_size,
    epochs=20,
    callbacks=[checkpoint]
)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# 训练和验证集的 loss 曲线
ax[0].plot(history.history['loss'], label='training loss')
ax[0].plot(history.history['val_loss'], label='validation loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training and Validation Loss')
ax[0].set_xticks(range(0, len(history.history['loss'])+1, 1))
ax[0].legend()

# 训练和验证集的 accuracy 曲线
ax[1].plot(history.history['accuracy'], label='training accuracy')
ax[1].plot(history.history['val_accuracy'], label='validation accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Training and Validation Accuracy')
ax[1].set_xticks(range(0, len(history.history['accuracy'])+1, 1))
ax[1].legend()

# 保存图片
plt.savefig('loss_acc.jpg')

#plt.show()


# 获取测试数据集的预测结果
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# 获取真实标签
true_labels = test_data.labels

# 打印分类报告
class_names = list(test_data.class_indices.keys())
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:")
print(report)


