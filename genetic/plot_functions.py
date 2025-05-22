import matplotlib.pyplot as plt
import numpy as np
import math

def p_delta_histograms(results: list, p_deltas: list, path: str):
    colors = ['blue', 'orange', 'green', 'red', 'purple']
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
    plt.close()

def p_delta_line(results: list, p_deltas: list, path: str):
    color='purple'
    results = np.array(results)
    p_deltas = np.array(p_deltas)

    mean_vals = np.mean(results, axis=1)
    std_vals = np.std(results, axis=1)

    plt.figure(figsize=(10, 6))
    plt.errorbar(p_deltas, mean_vals, yerr=std_vals,color=color, fmt='-o', capsize=5)
    plt.xlabel("P_delta Values")
    plt.ylabel("Average Generations to Reach Target")
    plt.title("Average Convergence Generations for Different Crossover Probabilities")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def generation_progress(bests, means, path: str):
    color=['purple', 'orange']
    generations = list(range(len(bests)))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, bests, label="Best fitness", color=color[0])
    plt.plot(generations, means, label="Mean fitness", color=color[1])
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value (log scale)")
    plt.title("Genetic Algorithm Progress Over Generations")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def multi_generation_progress(bests, means, labels, path: str):
    num_runs = len(bests)
    generations = [list(range(len(b))) for b in bests]
    colors = [
        ('purple', 'magenta'),
        ('saddlebrown', 'orange'),
        ('green', 'lime'),
        ('darkred', 'red'),
        ('teal', 'aqua'),
        ('navy', 'blue'),
    ]

    cols = math.ceil(math.sqrt(num_runs))
    rows = math.ceil(num_runs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(num_runs):
        ax = axes[i]
        color1, color2 = colors[i % len(colors)]
        ax.plot(generations[i], bests[i], label="Best fitness", color=color2)
        ax.plot(generations[i], means[i], label="Mean fitness", color=color1)
        ax.set_yscale('log')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (log scale)")
        ax.set_title(f"Run {i + 1}: {labels[i]}")
        ax.grid(True)
        ax.legend()

    for j in range(num_runs, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def schema_count(counts: list, labels: list, path: str):
    generations = list(range(len(counts)))
    plt.figure(figsize=(12, 8))

    for s in range(len(counts[0])):
        s_counts= [c[s] for c in counts]
        label = labels[s]
        plt.plot(generations, s_counts, label=label)

    plt.xlabel("Generation")
    plt.ylabel("Appearance Count")
    plt.title("Genetic Algorithm Schema Appearance Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()