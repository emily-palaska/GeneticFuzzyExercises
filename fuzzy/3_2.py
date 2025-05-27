from fuzzy.operations import plot_fuzzy, fuzzy_div, fuzzy_sub, fuzzy_mult

# Membership functions
def mu_A(x):
    if x <= 8: return 0
    if 8 < x < 18: return max(0, (1/10) * x - 8/10)
    if 18 <= x <= 32: return max(0, -(1/14) * x + 32/14)
    return 0

def mu_B(x):
    if x <= -3: return 0
    if -3 < x <= 6: return max(0, (1/9) * x - 1/3)
    if 6 < x <= 24: return max(0, -(1/18) * x + 3/4)
    return 0

# Domain definition
def domain(min_val, max_val, step):
    return [min_val + i*step for i in range(int((max_val - min_val)/step) + 1)]

x_A = domain(7, 33, 0.05)
x_B = domain(-4, 25, 0.05)

# Fuzzy numbers initialization
A = {x: mu_A(x) for x in x_A}
B = {x: mu_B(x) for x in x_B}

# (i) Plot μ_A and μ_B
plot_fuzzy(A, "Fuzzy Number A", '../results/3_2_a.png')
plot_fuzzy(B, "Fuzzy Number B", '../results/3_2_b.png')

# (ii) Fuzzy multiplication using extension principle (max-min convolution)
def filter_fuzzy(fuzzy_dict):
    items = list(fuzzy_dict.items())

    # Find first and last index where μ ≠ 0
    start = next((i for i, (_, mu) in enumerate(items) if mu > 0), None)
    end = next((i for i, (_, mu) in reversed(list(enumerate(items))) if mu > 0), None)

    if start is None or end is None: return {}

    # Keep only entries between first and last non-zero μ
    filtered_items = items[start:end + 1]
    return dict(filtered_items)

C = filter_fuzzy(fuzzy_mult(A, B))
plot_fuzzy(C, title="Fuzzy Product C = A * B", filename='../results/3_2_mult.png')

# (iii) Fuzzy subtraction and division using similar principle
D = filter_fuzzy(fuzzy_sub(A, B))
plot_fuzzy(D, title="Fuzzy Difference D = A - B", filename='../results/3_2_sub.png')
E = filter_fuzzy(fuzzy_mult(A, B))
plot_fuzzy(fuzzy_div(A, B), title="Fuzzy Division E = A / B", filename='../results/3_2_div.png')


