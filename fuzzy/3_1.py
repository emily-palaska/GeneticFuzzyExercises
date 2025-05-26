import torch
import matplotlib.pyplot as plt

# Define fuzzy numbers A and B as dictionaries
A_dict = {6: 0.33, 7: 0.67, 8: 1.00, 9: 0.67, 10: 0.33}
B_dict = {1: 0.33, 2: 0.67, 3: 1.00, 4: 0.67, 5: 0.33}

# Convert to PyTorch tensors (x: support, μ: membership)
def fuzzy_to_tensor(fuzzy_dict):
    x = torch.tensor(list(fuzzy_dict.keys()), dtype=torch.float32)
    mu = torch.tensor(list(fuzzy_dict.values()), dtype=torch.float32)
    return x, mu

A_x, A_mu = fuzzy_to_tensor(A_dict)
B_x, B_mu = fuzzy_to_tensor(B_dict)

# (i) Plot A and B
def plot_fuzzy(x, mu, title, filename):
    plt.stem(x.numpy(), mu.numpy())
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Membership")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_fuzzy(A_x, A_mu, "Fuzzy Number A", '../results/3_1_a.png')
plot_fuzzy(B_x, B_mu, "Fuzzy Number B", '../results/3_1_b.png')

# (ii) Fuzzy multiplication using extension principle (max-min convolution)
def fuzzy_multiply(x1, mu1, x2, mu2):
    result_dict = {}
    for i in range(len(x1)):
        for j in range(len(x2)):
            z = x1[i] * x2[j]
            mu = torch.minimum(mu1[i], mu2[j])
            z_rounded = round(z.item(), 2)  # Round for float stability
            # Use max-min aggregation
            if z_rounded in result_dict:
                result_dict[z_rounded] = max(result_dict[z_rounded], mu.item())
            else:
                result_dict[z_rounded] = mu.item()
    z_sorted = sorted(result_dict.keys())
    mu_sorted = [result_dict[z] for z in z_sorted]
    return torch.tensor(z_sorted), torch.tensor(mu_sorted)

C_x, C_mu = fuzzy_multiply(A_x, A_mu, B_x, B_mu)
plot_fuzzy(C_x, C_mu, "Fuzzy Product C = A * B", '../results/3_1_mult.png')

# (iii) Fuzzy subtraction and addition using similar principle
def fuzzy_sum(x1, mu1, x2, mu2):
    result_dict = {}
    for i in range(len(x1)):
        for j in range(len(x2)):
            z = x1[i] + x2[j]

            mu = min(mu1[i], mu2[j])
            z = round(z.item(), 2)
            result_dict[z] = max(result_dict.get(z, 0), mu)
    z_vals = torch.tensor(sorted(result_dict.keys()))
    mu_vals = torch.tensor([result_dict[z.item()] for z in z_vals])
    return z_vals, mu_vals

D_x, D_mu = fuzzy_sum(A_x, A_mu, -B_x, B_mu)
plot_fuzzy(D_x, D_mu, "Fuzzy Difference D = A - B", '../results/3_1_sub.png')

E_x, E_mu = fuzzy_sum(A_x, A_mu, B_x, B_mu)
plot_fuzzy(E_x, E_mu, "Fuzzy Sum E = A + B", '../results/3_1_sum.png')
