# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
    Y_hat = 1 / (1 + np.exp(-X @ W))  # 使用 sigmoid 函数进行预测
    G = (1/n) * X.T @ (Y_hat - Y)  # 梯度计算
    j = -(1/n) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))  # 代价函数 (交叉熵损失)
    
    return (j, G)

def gradientDescent(W, X, Y, n, lr, iterations):
    J = np.zeros([iterations, 1])
    
    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)
        W = W - lr * G  # 梯度下降更新 W

    return (W, J)

def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X @ W))  # 用 sigmoid 函数获取预测值
    Y_hat = (Y_hat >= 0.5).astype(int)  # 将概率值转换为二分类标签
    
    return (1 - np.mean(np.equal(Y_hat, Y)))

iterations = 1000  # 训练迭代次数
lr = 0.000125  # 学习率

data = np.loadtxt('D:\Machine Learning Lab\LR1.txt', delimiter=',')

n = data.shape[0]
W = np.random.random([3, 1])  # 随机初始化权重
X = np.concatenate([np.ones([n, 1]), data[:,0:2]], axis=1)  # 增加偏置项
Y = np.expand_dims(data[:, 2], axis=1)

(W, J) = gradientDescent(W, X, Y, n, lr, iterations)  # 使用梯度下降优化权重
print(error(W, X, Y))

# 绘制数据点和决策边界
idx0 = (data[:, 2] == 0)
idx1 = (data[:, 2] == 1)

plt.figure()
plt.ylim(-12, 12)
plt.plot(data[idx0, 0], data[idx0, 1], 'go')
plt.plot(data[idx1, 0], data[idx1, 1], 'rx')

x1 = np.arange(-10, 10, 0.2)
y1 = (W[0] + W[1] * x1) / -W[2]
plt.plot(x1, y1)

plt.figure()
plt.plot(range(iterations), J)
plt.show()  # 添加这行代码来显示图像