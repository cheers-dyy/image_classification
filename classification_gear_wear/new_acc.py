import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD


# 加载和处理数据的通用函数
def load_and_process_data(file_dir, size):
    data = scipy.io.loadmat(file_dir).get('X_DE_time').transpose()[0]
    length = len(data)
    length -= length % size
    data = data[:length]
    data = data.reshape(length // size, size)
    return data

# 读取训练数据
Normal_DIR = "./B118/N97.mat"
B007_DIR = "./B118/B118.mat"
IR007_DIR = "./B118/I105.mat"
OR007_DIR = "./B118/O130.mat"

Normal = load_and_process_data(Normal_DIR, 500)
Inner = load_and_process_data(IR007_DIR, 500)
Outer = load_and_process_data(OR007_DIR, 500)
Ball = load_and_process_data(B007_DIR, 500)

# 创建标签
labels = np.array([[1,0,0,0]] * (len(Normal)//500) +
                  [[0,1,0,0]] * (len(Inner)//500) +
                  [[0,0,1,0]] * (len(Outer)//500) +
                  [[0,0,0,1]] * (len(Ball)//500))

# 打乱顺序
data = np.concatenate([Normal, Inner, Outer, Ball])
indices = np.random.permutation(len(data))
data = data[indices]
labels = labels[indices]

# 归一化数据
mean = np.mean(data)
std = np.std(data)
data -= mean
data /= std

# 读取和处理测试数据
Normal_test = "./B225/N100.mat"
B021_test = "./B225/B225.mat"
IR021_test = "./B225/I212.mat"
OR021_test = "./B225/O237.mat"

Normaltest = load_and_process_data(Normal_test, 500)
Innertest = load_and_process_data(IR021_test, 500)
Outertest = load_and_process_data(OR021_test, 500)
Balltest = load_and_process_data(B021_test, 500)

# 创建测试标签
test_labels = np.array([[1,0,0,0]] * (len(Normaltest)//500) +
                       [[0,1,0,0]] * (len(Innertest)//500) +
                       [[0,0,1,0]] * (len(Outertest)//500) +
                       [[0,0,0,1]] * (len(Balltest)//500))

# 归一化测试数据
test_data = np.concatenate([Normaltest, Innertest, Outertest, Balltest])
test_data -= mean
test_data /= std

num_classes = 4
BatchNorm = True

# 实例化序贯模型
model = Sequential()

# 搭建输入层，第一层卷积。因为要指定input_shape，所以单独放出来
model.add(Conv1D(filters=8, kernel_size=16, strides=16, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4))

# 第二层卷积
model.add(Conv1D(filters=4, kernel_size=8, strides=2, padding='same',activation=tf.nn.relu))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(rate=0.3))

# 从卷积到全连接需要展平
model.add(Flatten())

# 添加全连接层
model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(1e-4)))

# 增加输出层
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))
model.summary()

# 定义SGD优化器
sgd = SGD(lr=0.001, momentum=0.9)  # 设置学习率和动量（可调整参数）

#定义优化器为Adam，损失函数为交叉熵损失
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

#模型训练，返回history对象，history为字典类型，包含val_loss,val_acc,loss,acc四个key值
#history=model.fit(trainData, trainLabel, batch_size=32, epochs=20, validation_split=0.2, shuffle= True)

history=model.fit(data, labels, batch_size=32, epochs=100, validation_data=(test_data, test_labels), shuffle= True)


#模型评估
loss,accuracy = model.evaluate(test_data, test_labels)

print('loss:',loss,'accuracy:', accuracy)


# 训练结果可视化
loss = history.history["loss"]
val_loss = history.history["val_loss"]
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.subplot(1,2,1)
plt.plot(loss,label = "Training Loss")
plt.plot(val_loss,label = "Validation Loss")
plt.title("Trainning and Validation Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(acc,label = "Training Acc")
plt.plot(val_acc,label = "Validation Acc")
plt.title("Training and Validation Acc")
plt.legend()

