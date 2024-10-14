import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

# -----------------------------
# 1. 生成数据集
# -----------------------------

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

# -----------------------------
# 2. 正确模型的最小期望风险分类器
# -----------------------------

# 计算所有样本的似然（正确模型）
p_x_given_L0 = multivariate_normal.pdf(X, mean=mu_0, cov=Sigma_0)
p_x_given_L1 = multivariate_normal.pdf(X, mean=mu_1, cov=Sigma_1)

# 计算似然比
likelihood_ratio = p_x_given_L1 / p_x_given_L0

# 定义 gamma 值
gamma_values = np.logspace(-3, 3, num=1000)

# 存储结果的列表
TPR_list = []
FPR_list = []
FNR_list = []
P_error_list = []

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

# 转换为 numpy 数组
gamma_values = np.array(gamma_values)
TPR_list = np.array(TPR_list)
FPR_list = np.array(FPR_list)
P_error_list = np.array(P_error_list)

# 找到最小错误概率
min_error_index = np.argmin(P_error_list)
P_min_error = P_error_list[min_error_index]
gamma_min = gamma_values[min_error_index]
TPR_min = TPR_list[min_error_index]
FPR_min = FPR_list[min_error_index]

# 计算理论最优阈值
gamma_theoretical = P_L0 / P_L1

# 找到理论最优阈值对应的索引
index_theoretical = np.argmin(np.abs(gamma_values - gamma_theoretical))
TPR_theoretical = TPR_list[index_theoretical]
FPR_theoretical = FPR_list[index_theoretical]

# -----------------------------
# 3. 朴素贝叶斯分类器的实现
# -----------------------------

# 提取对角协方差矩阵
Sigma_0_diag = np.diag(np.diag(Sigma_0))
Sigma_1_diag = np.diag(np.diag(Sigma_1))

# 计算所有样本的似然（朴素贝叶斯假设）
p_x_given_L0_NB = multivariate_normal.pdf(X, mean=mu_0, cov=Sigma_0_diag)
p_x_given_L1_NB = multivariate_normal.pdf(X, mean=mu_1, cov=Sigma_1_diag)

# 计算似然比
likelihood_ratio_NB = p_x_given_L1_NB / p_x_given_L0_NB

# 存储结果的列表
TPR_list_NB = []
FPR_list_NB = []
FNR_list_NB = []
P_error_list_NB = []

for gamma in gamma_values:
    D_NB = (likelihood_ratio_NB > gamma).astype(int)
    
    TP_NB = np.sum((D_NB == 1) & (L == 1))
    FP_NB = np.sum((D_NB == 1) & (L == 0))
    FN_NB = np.sum((D_NB == 0) & (L == 1))
    TN_NB = np.sum((D_NB == 0) & (L == 0))
    
    TPR_NB = TP_NB / (TP_NB + FN_NB)
    FPR_NB = FP_NB / (FP_NB + TN_NB)
    FNR_NB = FN_NB / (TP_NB + FN_NB)
    
    P_error_NB = FPR_NB * P_L0 + FNR_NB * P_L1
    
    TPR_list_NB.append(TPR_NB)
    FPR_list_NB.append(FPR_NB)
    FNR_list_NB.append(FNR_NB)
    P_error_list_NB.append(P_error_NB)

# 转换为 numpy 数组
TPR_list_NB = np.array(TPR_list_NB)
FPR_list_NB = np.array(FPR_list_NB)
P_error_list_NB = np.array(P_error_list_NB)

# 找到最小错误概率
min_error_index_NB = np.argmin(P_error_list_NB)
P_min_error_NB = P_error_list_NB[min_error_index_NB]
gamma_min_NB = gamma_values[min_error_index_NB]
TPR_min_NB = TPR_list_NB[min_error_index_NB]
FPR_min_NB = FPR_list_NB[min_error_index_NB]

# -----------------------------
# 4. 绘制 ROC 曲线并比较
# -----------------------------

plt.figure(figsize=(8,6))
plt.plot(FPR_list, TPR_list, label='Original ROC')
plt.plot(FPR_list_NB, TPR_list_NB, label='NB ROC', linestyle='--')

# 标记正确模型的最小错误概率点
plt.scatter(FPR_min, TPR_min, color='blue', marker='o', label='Min(Original)')

# 标记朴素贝叶斯的最小错误概率点
plt.scatter(FPR_min_NB, TPR_min_NB, color='red', marker='x', s=100, label='Min(NB)')

# 标记理论最优阈值点
plt.scatter(FPR_theoretical, TPR_theoretical, color='green', marker='s', s=100, label='Theoratical')

plt.xlabel('FPR (P(D=1|L=0;γ))')
plt.ylabel('TPR (P(D=1|L=1;γ))')
plt.title('ROC curve comparison')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 5. 打印结果和分析
# -----------------------------

print("===== Original Model =====")
print(f"γ_min: {gamma_min}")
print(f"P_min_error: {P_min_error}")
print(f"TPR: {TPR_min}")
print(f"FPR: {FPR_min}")

print("\n===== Theoratical =====")
print(f"γ_theoretical: {gamma_theoretical}")
print(f"TPR: {TPR_theoretical}")
print(f"FPR: {FPR_theoretical}")

print("\n===== NB Classifier =====")
print(f"P_min_error_NB: {P_min_error_NB}")
print(f"γ_min_NB: {gamma_min_NB}")
print(f"TPR: {TPR_min_NB}")
print(f"FPR: {FPR_min_NB}")

# 比较经验阈值与理论阈值
difference = abs(gamma_min - gamma_theoretical)
print(f"\nγ_min & γ_theoretical difference: {difference}")

# -----------------------------
# 6. 分析模型不匹配的影响
# -----------------------------

print("\n===== Impact of model mismatch =====")
print(f"P_min_error: {P_min_error}")
print(f"P_min_error_NB: {P_min_error_NB}")
print(f"Difference: {P_min_error_NB - P_min_error}")

# 绘制错误概率与Gamma的关系图
plt.figure(figsize=(10, 6))
plt.plot(gamma_values, P_error_list_NB, label='Errors')
plt.scatter(gamma_min_NB, P_min_error_NB, color='red', marker='o', label='Minimum Error')
plt.text(gamma_min_NB, P_min_error_NB, f'({gamma_min_NB:.4f}, {P_min_error_NB:.4f})', fontsize=10, ha='right', color='red')
plt.xscale('symlog')
plt.xlabel('Gamma')
plt.ylabel('Proportion of Errors')
plt.title('NB Probability of Error vs. Gamma')
plt.legend()
plt.grid(True)
plt.show()