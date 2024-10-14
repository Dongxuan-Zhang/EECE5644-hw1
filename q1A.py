import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

# Load saved samples and parameters
data = np.load('samples_and_labels.npz')
X = data['X']
L = data['L']
mu_0 = data['mu_0']
Sigma_0 = data['Sigma_0']
mu_1 = data['mu_1']
Sigma_1 = data['Sigma_1']
P_L0 = data['P_L0']
P_L1 = data['P_L1']

# Calculate likelihoods
p_x_given_L0 = multivariate_normal.pdf(X, mean=mu_0, cov=Sigma_0)
p_x_given_L1 = multivariate_normal.pdf(X, mean=mu_1, cov=Sigma_1)

# Calculate likelihood ratio
likelihood_ratio = p_x_given_L1 / p_x_given_L0

# Define gamma values
gamma_values = np.logspace(-3, 3, num=10000)

# Lists to store results
TPR_list = []
FPR_list = []
FNR_list = []
P_error_list = []

# ... code continues ...

# Find minimum error probability
min_error_index = np.argmin(P_error_list)
P_min_error = P_error_list[min_error_index]
gamma_min = gamma_values[min_error_index]
TPR_min = TPR_list[min_error_index]
FPR_min = FPR_list[min_error_index]

# Calculate theoretical optimal threshold
gamma_theoretical = P_L0 / P_L1
index_theoretical = np.argmin(np.abs(gamma_values - gamma_theoretical))
TPR_theoretical = TPR_list[index_theoretical]
FPR_theoretical = FPR_list[index_theoretical]

# Print results
print(f"γ_min: {gamma_min}")
print(f"P_min_error: {P_min_error}")
print(f"TPR_min: {TPR_min}")
print(f"FPR_min: {FPR_min}")

# Calculate theoretical optimal threshold
print(f"γ_theoretical: {gamma_theoretical}")

# Compare
difference = abs(gamma_min - gamma_theoretical)
print(f"Difference between γ_min and γ_theoretical: {difference}")

# Create a figure with two subplots
fig = plt.figure(figsize=(20, 8))

# Create 3D scatter plot
ax1 = fig.add_subplot(121, projection='3d')

# Create different scatter plots for L=0 and L=1
scatter0 = ax1.scatter(X[L==0, 0], X[L==0, 1], X[L==0, 2], c=X[L==0, 3], cmap='coolwarm', marker='o', label='L=0')
scatter1 = ax1.scatter(X[L==1, 0], X[L==1, 1], X[L==1, 2], c=X[L==1, 3], cmap='coolwarm', marker='^', label='L=1')

# Set axis labels
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Add colorbar
cbar = plt.colorbar(scatter0)
cbar.set_label('Dimension 4')

# Set title
ax1.set_title('4D Data Visualization')
ax1.legend()

# Create ROC curve plot
ax2 = fig.add_subplot(122)

# Plot ROC curve and mark minimum error probability point
ax2.plot(FPR_list, TPR_list, label='ROC Curve')
ax2.scatter(FPR_min, TPR_min, color='red', marker='o', label='Minimum')
ax2.scatter(FPR_theoretical, TPR_theoretical, color='green', marker='*', label='Theoretical')

ax2.set_xlabel('FPR (P(D=1|L=0;γ))')
ax2.set_ylabel('TPR (P(D=1|L=1;γ))')
ax2.set_title('ERM Classifier ROC Curve')
ax2.legend()
ax2.grid(True)

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Plot error probability vs Gamma
plt.figure(figsize=(10, 6))
plt.plot(gamma_values, P_error_list, label='Error')
plt.scatter(gamma_min, P_min_error, color='red', marker='o', label='Minimum Error')
plt.text(gamma_min, P_min_error, f'({gamma_min:.4f}, {P_min_error:.4f})', fontsize=10, ha='right', color='red')
plt.xscale('symlog')
plt.xlabel('Gamma')
plt.ylabel('Error Rate')
plt.title('Error Probability vs. Gamma')
plt.legend()
plt.grid(True)
plt.show()
