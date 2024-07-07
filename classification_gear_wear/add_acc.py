import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


Normal_DIR = "B118/N97.mat"
B007_DIR = "B118/B118.mat"
IR007_DIR = "B118/I105.mat"
OR007_DIR = "B118/O130.mat"

Normal=scipy.io.loadmat(Normal_DIR)['X097_DE_time'].transpose()[0]
Inner=scipy.io.loadmat(IR007_DIR)['X105_DE_time'].transpose()[0]
Outer=scipy.io.loadmat(OR007_DIR)['X130_DE_time'].transpose()[0]
Ball=scipy.io.loadmat(B007_DIR)['X118_DE_time'].transpose()[0]


lengthNormal = len(Normal);
lengthInner = len(Inner);
lengthOuter = len(Outer);
lengthBall = len(Ball);
print(lengthNormal, lengthInner, lengthOuter, lengthBall)
# 243938 121265 121991 122571


#规范数据长度，并且是每size长度为一条数据
size = 500;
lengthNormal = lengthNormal - lengthNormal%size;
lengthInner = lengthInner - lengthInner%size;
lengthOuter = lengthOuter - lengthOuter%size;
lengthBall = lengthBall - lengthBall%size;
Normal = Normal[0:lengthNormal]
Inner = Inner[0:lengthInner]
Outer = Outer[0:lengthOuter]
Ball = Ball[0:lengthBall]
print(Normal.shape)
#连接数据集
data = []
data.extend(Normal);
data.extend(Inner);
data.extend(Outer);
data.extend(Ball);
lengthData = lengthNormal + lengthInner + lengthOuter + lengthBall;
data = np.array(data);
print(data.shape)
print('数据个数：',lengthData/size)

#将一维数组变成二维（lengthData/size * size）
data = data.reshape(int(lengthData/size),size);
print(data.shape)


def normalize_data(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    data = (data - data_mean) / data_std
    return data


normalized_traindata = np.zeros_like(data)
start_idx = 0
end_idx = lengthNormal // size

# 归一化Normal类别的数据
for i in range(start_idx, end_idx):
    normalized_traindata[i] = normalize_data(data[i])

start_idx = end_idx
end_idx += lengthInner // size

# 归一化Inner类别的数据
for i in range(start_idx, end_idx):
    normalized_traindata[i] = normalize_data(data[i])

start_idx = end_idx
end_idx += lengthOuter // size

# 归一化Outer类别的数据
for i in range(start_idx, end_idx):
    normalized_traindata[i] = normalize_data(data[i])

start_idx = end_idx
# 归一化Ball类别的数据
for i in range(start_idx, len(normalized_traindata)):
    normalized_traindata[i] = normalize_data(data[i])

print('归一化后的测试数据:', normalized_traindata)

#在机器学习中，为了解决分类器不易处理离散数据的问题, 将标签改为one-hot编码形式
label = []
label.extend([[1,0,0,0] for i in range(0,int(lengthNormal/size))]);
label.extend([[0,1,0,0] for i in range(0,int(lengthInner/size))]);
label.extend([[0,0,1,0] for i in range(0,int(lengthOuter/size))]);
label.extend([[0,0,0,1] for i in range(0,int(lengthBall/size))]);
label = np.array(label)
print('label:',label)
print(label.shape)



#打乱顺序
index = [i for i in range(0,int(lengthData/size))];
np.random.shuffle(index);
randData = [];
randLabel = [];
for i in range(0,int(lengthData/size)):
    randData.append(data[index[i]]);
    randLabel.append(label[index[i]]);
randData = np.array(randData);
randLabel = np.array(randLabel);

trainData=randData;
trainLabel=randLabel;


#制作测试数据集
Normal_test = "B225/N100.mat"
B021_test = "B225/B225.mat"
IR021_test = "B225/I212.mat"
OR021_test = "B225/O237.mat"
Normaltest=scipy.io.loadmat(Normal_test)['X100_DE_time'].transpose()[0]
Innertest=scipy.io.loadmat(IR021_test)['X212_DE_time'].transpose()[0]
Outertest=scipy.io.loadmat(OR021_test)['X237_DE_time'].transpose()[0]
Balltest=scipy.io.loadmat(B021_test)['X225_DE_time'].transpose()[0]

#规范数据长度，并且是每size长度为一条数据
lengthNormal = len(Normaltest);
lengthInner = len(Innertest);
lengthOuter = len(Outertest);
lengthBall = len(Balltest);
size = 500;
lengthNormal = lengthNormal - lengthNormal%size;
lengthInner = lengthInner - lengthInner%size;
lengthOuter = lengthOuter - lengthOuter%size;
lengthBall = lengthBall - lengthBall%size;
Normaltest = Normaltest[0:lengthNormal]
Innertest = Innertest[0:lengthInner]
Outertest = Outertest[0:lengthOuter]
Balltest = Balltest[0:lengthBall]
print(Normaltest.shape)
#连接数据集
testdata = []
testdata.extend(Normaltest);
testdata.extend(Innertest);
testdata.extend(Outertest);
testdata.extend(Balltest);
lengthtestData = lengthNormal + lengthInner + lengthOuter + lengthBall;
testdata = np.array(testdata);
print(testdata.shape)
print('数据个数：',lengthtestData/size)

#将一维数组变成二维（lengthtestData/size * size）
testdata = testdata.reshape(int(lengthtestData/size),size);
print(testdata.shape)

# 归一化每个类别的数据
def normalize_data(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    data = (data - data_mean) / data_std
    return data

normalized_testdata = np.zeros_like(testdata)
start_idx = 0
end_idx = lengthNormal // size

# 归一化Normal类别的数据
for i in range(start_idx, end_idx):
    normalized_testdata[i] = normalize_data(testdata[i])

start_idx = end_idx
end_idx += lengthInner // size

# 归一化Inner类别的数据
for i in range(start_idx, end_idx):
    normalized_testdata[i] = normalize_data(testdata[i])

start_idx = end_idx
end_idx += lengthOuter // size

# 归一化Outer类别的数据
for i in range(start_idx, end_idx):
    normalized_testdata[i] = normalize_data(testdata[i])

start_idx = end_idx

# 归一化Ball类别的数据
for i in range(start_idx, len(normalized_testdata)):
    normalized_testdata[i] = normalize_data(testdata[i])

print('归一化后的测试数据:', normalized_testdata)


trainData=randData;
trainLabel=randLabel;
#在机器学习中，为了解决分类器不易处理离散数据的问题, 将标签改为one-hot编码形式
testlabel = []
testlabel.extend([[1,0,0,0] for i in range(0,int(lengthNormal/size))]);
testlabel.extend([[0,1,0,0] for i in range(0,int(lengthInner/size))]);
testlabel.extend([[0,0,1,0] for i in range(0,int(lengthOuter/size))]);
testlabel.extend([[0,0,0,1] for i in range(0,int(lengthBall/size))]);
testLabel = np.array(testlabel)
print('testlabel:',testLabel)
print(testLabel.shape)

#为了方便输入一维神经网络，将训练和测试数据改为(样本数，样本长度，1)的格式
trainData = trainData[:,:,np.newaxis]
testData = testdata[:,:,np.newaxis]
print(trainData.shape)
print(testData.shape)
input_shape = trainData.shape[1:]


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
#定义优化器为Adam，损失函数为交叉熵损失
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#模型训练，返回history对象，history为字典类型，包含val_loss,val_acc,loss,acc四个key值
history=model.fit(trainData, trainLabel, batch_size=64, epochs=20, validation_split=0.2)

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

#模型评估
loss,accuracy = model.evaluate(testData,testLabel)
print('loss:',loss,'accuracy:',accuracy)


