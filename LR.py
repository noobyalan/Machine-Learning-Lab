# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
    # 计算假设函数 (H = X * W)
    H = np.dot(X, W)
    # 计算代价函数 (均方误差)
    j = (1/(2*n)) * np.sum(np.square(H - Y))
    # 计算梯度 (偏导数)
    G = (1/n) * np.dot(X.T, (H - Y))
    return (j, G)

def gradientDescent(W, X, Y, lr, iterations):
    n = np.size(Y)  # 样本数量
    J = np.zeros([iterations, 1])  # 初始化存储每次迭代的代价值
    
    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)  # 计算当前迭代的代价和梯度
        W = W - lr * G  # 使用梯度下降更新权重 W
    
    return (W, J)

iterations = 1500  # 迭代次数
lr = 0.0001  # 学习率

data = np.loadtxt('LR.txt', delimiter=',')  # 从文件加载数据

n = np.size(data[:, 1])  # 样本数量
W = np.zeros([2, 1])  # 初始化权重 W
X = np.c_[np.ones([n, 1]), data[:,0]]  # 为 X 添加截距项 (第一列全为 1)
Y = data[:, 1].reshape([n, 1])  # 重塑 Y 的维度以匹配 X

(W, J) = gradientDescent(W, X, Y, lr, iterations)  # 调用梯度下降函数

# 绘制散点图和回归直线
plt.figure()
plt.plot(data[:,0], data[:,1], 'rx')  # 绘制数据点 (红色交叉)
plt.plot(data[:,0], np.dot(X, W))  # 绘制拟合直线

# 绘制代价函数随迭代变化的图
plt.figure()
plt.plot(range(iterations), J)  # 绘制代价函数曲线
plt.show()
