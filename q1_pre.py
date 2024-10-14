import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 定义类别先验概率
P_L0 = 0.35
P_L1 = 0.65

# 总样本数
N = 10000

# 每个类别的样本数
N_L0 = int(N * P_L0)
N_L1 = N - N_L0

# 定义每个类别的均值和协方差
mu_0 = np.array([-1, -1, -1, -1])  # 用实际值替换
Sigma_0 = np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]])

mu_1 = np.array([1, 1, 1, 1])  # 用实际值替换
Sigma_1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])

# 生成样本
X_L0 = np.random.multivariate_normal(mu_0, Sigma_0, N_L0)
X_L1 = np.random.multivariate_normal(mu_1, Sigma_1, N_L1)

# 合并样本和标签
X = np.vstack((X_L0, X_L1))
L = np.hstack((np.zeros(N_L0), np.ones(N_L1)))

# 保存样本和标签
np.savez('samples_and_labels.npz', X=X, L=L, mu_0=mu_0, Sigma_0=Sigma_0, mu_1=mu_1, Sigma_1=Sigma_1, P_L0=P_L0, P_L1=P_L1)

print("save to: samples_and_labels.npz")