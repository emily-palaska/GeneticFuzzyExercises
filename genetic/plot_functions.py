import matplotlib.pyplot as plt
import numpy as np

def p_delta_histograms(results: list, p_deltas: list, path: str):
    colors = ['blue', 'orange']
    plt.figure(figsize=(10, 6))

    for i, (gen_data, p_delta) in enumerate(zip(results, p_deltas)):
        plt.hist(gen_data, bins=10, alpha=0.6, color=colors[i], label=f"P_delta = {p_delta}")
        mean_val = np.mean(gen_data)
        plt.axvline(mean_val, color=colors[i], linestyle='--', linewidth=2,
                    label=f"Mean (P_delta = {p_delta}): {mean_val:.2f}")

    plt.xlabel("Generations to Reach Target")
    plt.ylabel("Frequency")
    plt.title("Histogram of Generations to Reach Target for Different Crossover Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)