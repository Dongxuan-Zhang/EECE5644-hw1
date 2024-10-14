import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据集
dataset = np.load('dataset_Q2.npz')
data = dataset['data']
labels = dataset['labels']

# 样本总数
num_samples = len(data)

# 定义均值向量
mu1 = np.array([0, 0, 0])
mu2 = np.array([2.5, 0, 0])
mu3 = np.array([0, 2.5, 0])
mu4 = np.array([2.5, 2.5, 0])

# 定义先验概率
P_w1 = 0.3
P_w2 = 0.4
P_w3 = 0.3

# 最小错误概率的贝叶斯决策规则（使用0-1损失）
def decision_rule_bayes(x):
    # 计算属于每个类别的条件概率密度
    p1 = np.exp(-0.5 * np.sum((x - mu1) ** 2)) * P_w1
    p2 = np.exp(-0.5 * np.sum((x - mu2) ** 2)) * P_w2
    p3 = 0.5 * (np.exp(-0.5 * np.sum((x - mu3) ** 2)) + np.exp(-0.5 * np.sum((x - mu4) ** 2))) * P_w3
    # 根据最大化后验概率来做决策
    return np.argmax([p1, p2, p3]) + 1

# 分类10K样本并统计每个决策-标签对
confusion_matrix = np.zeros((3, 3))
predictions = []
for i in range(num_samples):
    x = data[i]
    true_label = labels[i]
    predicted_label = decision_rule_bayes(x)
    predictions.append(predicted_label)
    confusion_matrix[predicted_label - 1, true_label - 1] += 1

print("confusion_matrix：")
print(confusion_matrix / num_samples)

# 三维散点图显示数据，用不同的形状和颜色区分正确与否
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建图例句柄
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Class 1 (Correct)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Class 1 (Incorrect)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10, label='Class 2 (Correct)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='r', markersize=10, label='Class 2 (Incorrect)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='g', markersize=10, label='Class 3 (Correct)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='r', markersize=10, label='Class 3 (Incorrect)')
]

# 绘制样本
for i in range(num_samples):
    x = data[i]
    true_label = labels[i]
    predicted_label = predictions[i]
    if true_label == predicted_label:
        color = 'g'  # 正确分类为绿色
    else:
        color = 'r'  # 错误分类为红色

    if true_label == 1:
        marker = 'o'
    elif true_label == 2:
        marker = '^'
    else:
        marker = 's'

    ax.scatter(x[0], x[1], x[2], c=color, marker=marker)

# 设置图例和标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.legend(handles=legend_elements, loc='upper right')
ax.set_title('3D Scatter Plot of Data Classification Results')
plt.show()