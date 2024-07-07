import numpy as np
import matplotlib.pyplot as plt

# 加载保存的loss值
model1_loss = np.load('cbam_losses.npy')
model2_loss = np.load('losses.npy')
model3_loss = np.load('se_losses.npy')

# 设置横坐标为50个epoch
epochs = np.arange(1, 51)

# 设置绘图参数
plt.figure(figsize=(8, 6))
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制曲线
plt.plot(epochs, model1_loss, label='Model 1')
plt.plot(epochs, model2_loss, label='Model 2')
plt.plot(epochs, model3_loss, label='Model 3')

# 添加图例并显示图形
plt.legend()
plt.show()