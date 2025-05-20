import numpy as np
import matplotlib.pyplot as plt

def fitness(chromosome: np.ndarray, target: np.ndarray):
    assert len(chromosome) == len(target), f"{len(chromosome)} != {len(target)}"
    return np.sum(chromosome == target)

def roulette_wheel_selection(population: np.ndarray, fitnesses: np.ndarray):
    assert len(population) == len(fitnesses), f'{len(population)} != {len(fitnesses)}'
    probabilities = fitnesses / np.sum(fitnesses)
    indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[indices[0]], population[indices[1]]

def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray, P_delta: float):
    if np.random.rand() < P_delta:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return np.copy(parent1), np.copy(parent2)

def mutate(chromosome: np.ndarray, P_mu: float):
    mutation_mask = np.random.rand(len(chromosome)) < P_mu
    chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
    return chromosome

def run_genetic_algorithm(P_delta: float, P_mu=0.001, n=100, l=100, max_generations=1e6):
    target = np.ones(l, dtype=int)
    population = np.random.randint(0, 2, size=(n, l))

    for generation in range(int(max_generations)):
        fitnesses = np.array([fitness(chromosome, target) for chromosome in population])
        if np.any(fitnesses == l):
            return generation

        new_population = []
        while len(new_population) < n:
            parent1, parent2 = roulette_wheel_selection(population, fitnesses)
            child1, child2 = one_point_crossover(parent1, parent2, P_delta)
            new_population.append(mutate(child1, P_mu))
            new_population.append(mutate(child2, P_mu))
        population = np.array(new_population[:n])

    return max_generations

def plot_results(results: list, p_deltas: list):
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
    plt.savefig('../results/9_1_target_histogram.png')


def main():
    runs, results, p_deltas = 20, [], [0.7, 0.0]
    for P_delta in p_deltas:
        generations = [run_genetic_algorithm(P_delta) for _ in range(runs)]
        results.append(generations)
        avg_generation = np.mean(generations)
        print(f"Average generation for P_delta = {P_delta}: {avg_generation:.2f}")
    plot_results(results, p_deltas)

if __name__ == "__main__":
    main()
