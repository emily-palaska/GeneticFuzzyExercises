import torch
import matplotlib.pyplot as plt

# Set grid resolution
grid_size = 200
x = torch.linspace(-2, 2, grid_size)
y = torch.linspace(-2, 2, grid_size)
X, Y = torch.meshgrid(x, y, indexing='ij')

# (a) x close to origin with y
mu_R_origin = torch.exp(- (X**2 + Y**2))

# (b) x close to perimeter of circle of radius 1 with y
mu_R_circle = torch.exp(-((X**2 + Y**2 - 1)**2 + Y**2))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].imshow(mu_R_origin.numpy(), extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
axs[0].set_title(r"Fuzzy relation: $x$ close to origin with $y$")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

axs[1].imshow(mu_R_circle.numpy(), extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
axs[1].set_title(r"Fuzzy relation: $x$ close to circle perimeter with $y$")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

plt.tight_layout()
plt.savefig('../results/2_8_fuzzy.png')
plt.close()
