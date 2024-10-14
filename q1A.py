import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

# 加载保存的样本和参数
data = np.load('samples_and_labels.npz')
X = data['X']
L = data['L']
mu_0 = data['mu_0']
Sigma_0 = data['Sigma_0']
mu_1 = data['mu_1']
Sigma_1 = data['Sigma_1']
P_L0 = data['P_L0']
P_L1 = data['P_L1']

# 计算似然
p_x_given_L0 = multivariate_normal.pdf(X, mean=mu_0, cov=Sigma_0)
p_x_given_L1 = multivariate_normal.pdf(X, mean=mu_1, cov=Sigma_1)

# 计算似然比
likelihood_ratio = p_x_given_L1 / p_x_given_L0

# 定义 gamma 值
gamma_values = np.logspace(-3, 3, num=10000)

# 存储结果的列表
TPR_list = []
FPR_list = []
FNR_list = []
P_error_list = []

P_L0 = 0.35
P_L1 = 0.65

for gamma in gamma_values:
    D = (likelihood_ratio > gamma).astype(int)
    
    TP = np.sum((D == 1) & (L == 1))
    FP = np.sum((D == 1) & (L == 0))
    FN = np.sum((D == 0) & (L == 1))
    TN = np.sum((D == 0) & (L == 0))
    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    
    P_error = FPR * P_L0 + FNR * P_L1
    
    TPR_list.append(TPR)
    FPR_list.append(FPR)
    FNR_list.append(FNR)
    P_error_list.append(P_error)

# 找到最小错误概率
min_error_index = np.argmin(P_error_list)
P_min_error = P_error_list[min_error_index]
gamma_min = gamma_values[min_error_index]
TPR_min = TPR_list[min_error_index]
FPR_min = FPR_list[min_error_index]

gamma_theoretical = P_L0 / P_L1
index_theoretical = np.argmin(np.abs(gamma_values - gamma_theoretical))
TPR_theoretical = TPR_list[index_theoretical]
FPR_theoretical = FPR_list[index_theoretical]

# 打印结果
print(f"γ_min: {gamma_min}")
print(f"P_min_error: {P_min_error}")
print(f"TPR_min: {TPR_min}")
print(f"FPR_min: {FPR_min}")

# 计算理论最优间值
print(f"γ_theoretical: {gamma_theoretical}")

# 比较
difference = abs(gamma_min - gamma_theoretical)
print(f"γ_min & γ_theoretical difference: {difference}")

# 创建一个包含两个子图的图形
fig = plt.figure(figsize=(20, 8))

# 创建3D散点图
ax1 = fig.add_subplot(121, projection='3d')

# 为L=0和L=1创建不同的散点图
scatter0 = ax1.scatter(X[L==0, 0], X[L==0, 1], X[L==0, 2], c=X[L==0, 3], cmap='coolwarm', marker='o', label='L=0')
scatter1 = ax1.scatter(X[L==1, 0], X[L==1, 1], X[L==1, 2], c=X[L==1, 3], cmap='coolwarm', marker='^', label='L=1')

# 设置坐标轴标签
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 添加颜色条
cbar = plt.colorbar(scatter0)
cbar.set_label('Dimension 4')

# 设置标题
ax1.set_title('4 dimensional data visualization')
ax1.legend()

# 创建ROC曲线图
ax2 = fig.add_subplot(122)

# 绘制 ROC 曲线并标记最小错误概率点
ax2.plot(FPR_list, TPR_list, label='ROC curve')
ax2.scatter(FPR_min, TPR_min, color='red', marker='o', label='Min')
ax2.scatter(FPR_theoretical, TPR_theoretical, color='green', marker='*', label='Theoretical')

ax2.set_xlabel('FPR (P(D=1|L=0;γ))')
ax2.set_ylabel('TPR (P(D=1|L=1;γ))')
ax2.set_title('ERM Classifier ROC Curve')
ax2.legend()
ax2.grid(True)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()

# 绘制错误概率与Gamma的关系图
plt.figure(figsize=(10, 6))
plt.plot(gamma_values, P_error_list, label='Errors')
plt.scatter(gamma_min, P_min_error, color='red', marker='o', label='Minimum Error')
plt.text(gamma_min, P_min_error, f'({gamma_min:.4f}, {P_min_error:.4f})', fontsize=10, ha='right', color='red')
plt.xscale('symlog')
plt.xlabel('Gamma')
plt.ylabel('Proportion of Errors')
plt.title('Probability of Error vs. Gamma')
plt.legend()
plt.grid(True)
plt.show()