from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('agg')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"



def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_pred_threshold = K.cast(K.greater(y_pred, 0.5), 'float32')  # 阈值化预测结果
    intersection = K.sum(y_true * y_pred_threshold)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred_threshold) + smooth)

# 自定义 IoU
def iou(y_true, y_pred):
    smooth = 1e-15
    y_pred_threshold = K.cast(K.greater(y_pred, 0.5), 'float32')  # 阈值化预测结果
    intersection = K.sum(K.abs(y_true * y_pred_threshold))
    union = K.sum(y_true) + K.sum(y_pred_threshold) - intersection
    return (intersection + smooth) / (union + smooth)


# 加载保存的模型
saved_model_path = "/home/dingying/fish/u_net/model_epoch_39_val_acc_0.9599.h5"  # 替换为实际保存的模型路径
loaded_model = load_model(saved_model_path, custom_objects={'dice_coefficient': dice_coefficient, 'iou': iou})

# 加载测试集数据
test_data_path = 'test_x.npy'  # 替换为保存测试集数据的.npy文件路径
test_data = np.load(test_data_path)


# 如果测试数据的格式需要调整，请按需进行调整
# 例如，如果测试数据是归一化的（0到1之间的浮点数），则不需要额外的预处理步骤
test_data = test_data.astype('float32') / 255.0

# 进行预测
predictions = loaded_model.predict(test_data)

# Threshold the predictions
threshold = 0.5
predictions[predictions > threshold] = 1
predictions[predictions <= threshold] = 0

np.save('predicƟons.npy', predictions)








