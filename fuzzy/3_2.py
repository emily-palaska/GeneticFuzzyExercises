import torch
import matplotlib.pyplot as plt

# Domain definition
x_A = torch.linspace(0, 40, 1000)
x_B = torch.linspace(-10, 30, 1000)

# Membership function for A
def mu_A(x):
    return torch.where(
        x <= 8, 0.0,
        torch.where(
            x < 18, 0.1 * x - 0.8,
            torch.where(x <= 32, -1/14 * x + 32/14, 0.0)
        )
    )

# Membership function for B
def mu_B(x):
    return torch.where(
        x <= -3, 0.0,
        torch.where(
            x <= 6, 1/9 * x - 1/3,
            torch.where(x <= 24, -1/18 * x + 4/3, 0.0)
        )
    )

def fuzzy_operation(xA, muA, xB, muB, op="add"):
    result = {}
    for i in range(len(xA)):
        for j in range(len(xB)):
            a = xA[i].item()
            b = xB[j].item()
            μ = min(muA[i].item(), muB[j].item())
            if op == "add":
                z = a + b
            elif op == "sub":
                z = a - b
            elif op == "mul":
                z = a * b
            z = round(z, 2)
            if z in result:
                result[z] = max(result[z], μ)
            else:
                result[z] = μ
    z_vals = sorted(result.keys())
    mu_vals = [result[z] for z in z_vals]
    return torch.tensor(z_vals), torch.tensor(mu_vals)

muA_vals = mu_A(x_A)
muB_vals = mu_B(x_B)

# Plot μ_A and μ_B
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_A, muA_vals)
plt.title("Membership Function μ_A(x)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_B, muB_vals)
plt.title("Membership Function μ_B(x)")
plt.grid(True)
plt.tight_layout()
plt.show()

C_x, C_mu = fuzzy_operation(x_A, muA_vals, x_B, muB_vals, op="mul")
D_x, D_mu = fuzzy_operation(x_A, muA_vals, x_B, muB_vals, op="sub")
E_x, E_mu = fuzzy_operation(x_A, muA_vals, x_B, muB_vals, op="add")

# Plot all results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(C_x, C_mu)
plt.title("Fuzzy Product: $C = A \\cdot B$")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(D_x, D_mu)
plt.title("Fuzzy Difference: $D = A - B$")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(E_x, E_mu)
plt.title("Fuzzy Sum: $E = A + B$")
plt.grid(True)

plt.tight_layout()
plt.show()

