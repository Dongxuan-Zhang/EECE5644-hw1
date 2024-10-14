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
mu1 = np.array([1, 1, 1])
mu2 = np.array([2, 2, 2])
mu3 = np.array([3, 3, 3])
mu4 = np.array([4, 4, 4])

# 定义损失矩阵
loss_matrix_10 = np.array([[0, 10, 10],
                           [1, 0, 10],
                           [1, 1, 0]])

loss_matrix_100 = np.array([[0, 100, 100],
                            [1, 0, 100],
                            [1, 1, 0]])

# 最小错误概率的决策规则（考虑损失矩阵）
def decision_rule(x, loss_matrix):
    # 计算属于每个类别的条件概率密度
    p1 = np.exp(-0.5 * np.sum((x - mu1) ** 2))
    p2 = np.exp(-0.5 * np.sum((x - mu2) ** 2))
    p3 = 0.5 * (np.exp(-0.5 * np.sum((x - mu3) ** 2)) + np.exp(-0.5 * np.sum((x - mu4) ** 2)))
    probs = np.array([p1, p2, p3])
    # 计算期望损失
    risks = loss_matrix @ probs
    # 返回最小风险对应的类别
    return np.argmin(risks) + 1

# 使用损失矩阵Λ_10分类10K样本并统计每个决策-标签对
confusion_matrix_10 = np.zeros((3, 3))
predictions_10 = []
for i in range(num_samples):
    x = data[i]
    true_label = labels[i]
    predicted_label = decision_rule(x, loss_matrix_10)
    predictions_10.append(predicted_label)
    confusion_matrix_10[predicted_label - 1, true_label - 1] += 1

print("confusion matrix(loss matrix Λ_10)：")
print(confusion_matrix_10 / num_samples)

# 使用损失矩阵Λ_100分类10K样本并统计每个决策-标签对
confusion_matrix_100 = np.zeros((3, 3))
predictions_100 = []
for i in range(num_samples):
    x = data[i]
    true_label = labels[i]
    predicted_label = decision_rule(x, loss_matrix_100)
    predictions_100.append(predicted_label)
    confusion_matrix_100[predicted_label - 1, true_label - 1] += 1

print("confusion matrix(loss matrix Λ_100)：")
print(confusion_matrix_100 / num_samples)

# 创建两个子图，分别显示使用Λ_10和Λ_100的分类结果
fig = plt.figure(figsize=(15, 6))

# 子图1：使用Λ_10的分类结果
ax1 = fig.add_subplot(121, projection='3d')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Class 1 (Correct)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Class 1 (Incorrect)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10, label='Class 2 (Correct)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='r', markersize=10, label='Class 2 (Incorrect)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='g', markersize=10, label='Class 3 (Correct)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='r', markersize=10, label='Class 3 (Incorrect)')
]

for i in range(num_samples):
    x = data[i]
    true_label = labels[i]
    predicted_label = predictions_10[i]
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

    ax1.scatter(x[0], x[1], x[2], c=color, marker=marker)

ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('X3')
ax1.set_title('3D Scatter Plot of Data Classification Results (Λ_10)')
ax1.legend(handles=legend_elements, loc='upper right')

# 子图2：使用Λ_100的分类结果
ax2 = fig.add_subplot(122, projection='3d')

for i in range(num_samples):
    x = data[i]
    true_label = labels[i]
    predicted_label = predictions_100[i]
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

    ax2.scatter(x[0], x[1], x[2], c=color, marker=marker)

ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('X3')
ax2.set_title('3D Scatter Plot of Data Classification Results (Λ_100)')
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()