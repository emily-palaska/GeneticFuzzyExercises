import torch
import matplotlib.pyplot as plt

# Define the grid
x1 = torch.linspace(0, 3, 300)
x2 = torch.linspace(0, 3, 300)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')

# Densities
p1 = torch.exp(-X1)
p2 = X2 * torch.exp(-X2)

# Fuzzy similarity membership
alpha = 1.0
mu = torch.exp(-alpha * (X1 - X2)**2)

# Fuzzy-weighted joint probability
prob_matrix = mu * p1 * p2

# Numerical integration
dx = (x1[1] - x1[0]).item()
dy = (x2[1] - x2[0]).item()
P = torch.sum(prob_matrix) * dx * dy
print(f"Fuzzy similarity probability: {P.item():.4f}")

# Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

X1_np = X1.numpy()
X2_np = X2.numpy()
Z_np = prob_matrix.numpy()

ax.plot_surface(X1_np, X2_np, Z_np, cmap='viridis', linewidth=0, antialiased=False)
ax.set_title('Fuzzy Weighted Joint Probability Surface')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel(r'$\mu(x_1, x_2) \cdot p_1(x_1) \cdot p_2(x_2)$')

plt.tight_layout()
plt.savefig('../results/2_12_fuzzy_3d.png')
plt.close()
