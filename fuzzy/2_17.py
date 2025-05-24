import torch
import matplotlib.pyplot as plt

def S(x: torch.Tensor, l: float, r: float):
    y = torch.zeros_like(x)

    mask1 = x <= l
    mask2 = (x > l) & (x < (l + r) / 2)
    mask3 = (x >= (l + r) / 2) & (x <= r)
    mask4 = x > r

    y[mask1] = 0
    y[mask2] = 2 * ((x[mask2] - l) / (r - l)) ** 2
    y[mask3] = 1 - 2 * ((r - x[mask3]) / (r - l)) ** 2
    y[mask4] = 1

    return y

def dS_dx(x: torch.Tensor, l: float, r: float):
    dy = torch.zeros_like(x)

    mask1 = (x <= l) | (x > r)
    mask2 = (x > l) & (x < (l + r) / 2)
    mask3 = (x >= (l + r) / 2) & (x <= r)

    dy[mask1] = 0
    dy[mask2] = 4 * (x[mask2] - l) / ((r - l) ** 2)
    dy[mask3] = 4 * (r - x[mask3]) / ((r - l) ** 2)

    return dy

def intersection_point(l: float, r: float):
    x_mid = (l + r) / 2
    y_mid = S(torch.tensor([x_mid]), l, r).item()
    return x_mid, y_mid

# Plotting S and its derivative for various l, r
x_vals = torch.linspace(0, 10, 500, dtype=torch.float32)
params = [(2, 6), (2, 7), (3, 6), (3, 7)]

plt.figure(figsize=(12, 6))

# Plot S(x; l, r)
for l, r in params:
    y_vals = S(x_vals, l, r)
    plt.plot(x_vals.numpy(), y_vals.numpy(), label=f'l={l}, r={r}')
plt.title('S-shaped Membership Function')
plt.xlabel('x')
plt.ylabel('S(x; l, r)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../results/2_17_sweep.png')
plt.close()

# Plot derivative dS/dx
for l, r in params:
    dy_vals = dS_dx(x_vals, l, r)
    plt.plot(x_vals.numpy(), dy_vals.numpy(), label=f'l={l}, r={r}')
plt.title('Derivative of S(x; l, r)')
plt.xlabel('x')
plt.ylabel(r"$\frac{d}{dx} S(x; l, r)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../results/2_17_derivative.png')
plt.close()

# Intersection points printout
for l, r in params:
    x_cross, y_cross = intersection_point(l, r)
    print(f'Intersection for l={l}, r={r}: x = {x_cross:.2f}, S = {y_cross:.2f}')
