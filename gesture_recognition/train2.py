

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


data_dir = "/root/aiFile/picture/gesture_data/"
labels = os.listdir(data_dir)
images = []
labels_list = []

for label in labels:
    path = os.path.join(data_dir, label)
    for image_file in os.listdir(path):
        image = cv2.imread(os.path.join(path, image_file), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (227, 227))
        # 假设image是您的图像数据，数据类型为uint8
        image = image.astype(np.float64)  # 将图像数据转换为float64类型

        image = image / 255.0
        images.append(image)
        labels_list.append(labels.index(label))

images = np.array(images).reshape(-1, 227, 227, 1)
labels = np.array(labels_list)

shuffled_images, shuffled_labels = shuffle(images, labels, random_state=42)

X_train, X_other, y_train, y_other = train_test_split(shuffled_images, shuffled_labels, test_size=0.4, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=64, shuffle=True)

# 数据增强
def data_augment(image):
    # 随机旋转
    angle = np.random.randint(-10, 10)
    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 随机缩放
    scale = np.random.uniform(0.9, 1.1)
    h, w = image.shape[:2]
    new_h, new_w = int(h*scale), int(w*scale)
    image = cv2.resize(image, (new_w, new_h))
    image = image[(new_h-h)//2:(new_h-h)//2+h, (new_w-w)//2:(new_w-w)//2+w]

    # 随机水平翻转
    if np.random.random() < 0.5:
        image = cv2.flip(image, 1)

    # 随机垂直翻转
    if np.random.random() < 0.5:
        image = cv2.flip(image, 0)

    # 随机剪裁
    crop_size = np.random.randint(0, 10)
    if crop_size > 0:
        image = image[crop_size:-crop_size, crop_size:-crop_size]

    return image

X_train_aug = []
y_train_aug = []

for i in range(X_train.shape[0]):
    image = X_train[i]
    label = y_train[i]

    # 数据增强
    image_aug = data_augment(image)
    X_train_aug.append(image_aug)
    y_train_aug.append(label)

X_train_aug = np.array(X_train_aug)
#X_train_aug = X_train_aug.reshape(-1, 227, 227, 1)
y_train_aug = np.array(y_train_aug)

#train_datagen = ImageDataGenerator()

y_train_aug = to_categorical(y_train_aug)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

model = Sequential([
    Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 1)),
    MaxPooling2D((3, 3), strides=(2, 2)),
    Conv2D(256, (5, 5), activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2)),
    Conv2D(384, (3, 3), activation='relu'),
    Conv2D(384, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(len(labels), activation='softmax')  # 输出维度改为类别数
])

model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = model.fit(X_train_aug, y_train_aug, batch_size=32, epochs=80, validation_data=(X_val, y_val), callbacks=[checkpoint], steps_per_epoch=len(X_train_aug)//32)

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
report = classification_report(np.argmax(y_test, axis=1), y_pred_labels)

print("指标报告：\n", report)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.jpg')

plt.figure()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc.jpg')
