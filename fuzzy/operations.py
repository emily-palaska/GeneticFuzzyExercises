import matplotlib.pyplot as plt

def unzip_fuzzy(fuzzy_number):
    x = list(fuzzy_number.keys())
    mu = list(fuzzy_number.values())
    return x, mu

def fuzzy_sum(a, b):
    result = {}
    for a_val, a_mu in a.items():
        for b_val, b_mu in b.items():
            sum_val = a_val + b_val
            mu = min(a_mu, b_mu)
            if sum_val in result:
                result[sum_val] = max(result[sum_val], mu)
            else:
                result[sum_val] = mu
    return dict(sorted(result.items()))

def fuzzy_sub(A, B):
    result = {}
    for a_val, a_mu in A.items():
        for b_val, b_mu in B.items():
            sub_val = a_val - b_val
            mu = min(a_mu, b_mu)
            if sub_val in result:
                result[sub_val] = max(result[sub_val], mu)
            else:
                result[sub_val] = mu
    return dict(sorted(result.items()))

def fuzzy_mult(A, B):
    result = {}
    for a_val, a_mu in A.items():
        for b_val, b_mu in B.items():
            mult_val = a_val * b_val
            mu = min(a_mu, b_mu)
            if mult_val in result:
                result[mult_val] = max(result[mult_val], mu)
            else:
                result[mult_val] = mu
    return dict(sorted(result.items()))

def fuzzy_div(A, B):
    result = {}
    for a_val, a_mu in A.items():
        for b_val, b_mu in B.items():
            if b_val != 0:
                div_val = a_val / b_val
                mu = min(a_mu, b_mu)
                if div_val in result:
                    result[div_val] = max(result[div_val], mu)
                else:
                    result[div_val] = mu
    return dict(sorted(result.items()))

def plot_fuzzy(fuzzy_number: dict, title: str, filename: str):
    x, mu = unzip_fuzzy(fuzzy_number)

    if len(x) < 100: plt.stem(x, mu)
    else: plt.plot(x, mu)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Membership")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
