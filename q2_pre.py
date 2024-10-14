import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子以保证结果可重复
np.random.seed(0)

# 样本总数
num_samples = 10000

# 定义类别先验概率
priors = [0.3, 0.3, 0.4]

# 计算每个类别的样本数量
num_samples_per_class = np.random.multinomial(num_samples, priors)

# 定义均值向量和协方差矩阵
mu1 = np.array([0, 0, 0])
mu2 = np.array([2.5, 0, 0])
mu3 = np.array([0, 2.5, 0])
mu4 = np.array([2.5, 2.5, 0])
cov = np.eye(3)  # 3x3单位矩阵

# 初始化数据和标签列表
data = []
labels = []

# 生成类别1的样本
samples_class1 = np.random.multivariate_normal(mu1, cov, num_samples_per_class[0])
data.append(samples_class1)
labels.extend([1] * num_samples_per_class[0])

# 生成类别2的样本
samples_class2 = np.random.multivariate_normal(mu2, cov, num_samples_per_class[1])
data.append(samples_class2)
labels.extend([2] * num_samples_per_class[1])

# 生成类别3的样本（来自两个高斯成分的等权混合）
samples_class3 = []
for _ in range(num_samples_per_class[2]):
    # 随机选择高斯成分
    component = np.random.choice([3, 4])
    if component == 3:
        sample = np.random.multivariate_normal(mu3, cov)
    else:
        sample = np.random.multivariate_normal(mu4, cov)
    samples_class3.append(sample)

data.append(np.array(samples_class3))
labels.extend([3] * num_samples_per_class[2])

# 将数据和标签合并
data = np.vstack(data)
labels = np.array(labels)

# 打乱数据顺序
indices = np.arange(num_samples)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# 合并数据和标签为一个数组
dataset = np.column_stack((data, labels))

# 输出数据维度以验证
print("shape: ", dataset.shape)  # 应为 (10000, 4)，前三列是数据，最后一列是标签

np.savez('dataset_q2.npz', data=data, labels=labels)

# 三维散点图显示数据，用不同的形状区分标签
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制类别1的样本
class1_indices = labels == 1
ax.scatter(data[class1_indices, 0], data[class1_indices, 1], data[class1_indices, 2], marker='o', label='Class 1')

# 绘制类别2的样本
class2_indices = labels == 2
ax.scatter(data[class2_indices, 0], data[class2_indices, 1], data[class2_indices, 2], marker='^', label='Class 2')

# 绘制类别3的样本
class3_indices = labels == 3
ax.scatter(data[class3_indices, 0], data[class3_indices, 1], data[class3_indices, 2], marker='s', label='Class 3')

# 设置图例和标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.legend()

plt.show()