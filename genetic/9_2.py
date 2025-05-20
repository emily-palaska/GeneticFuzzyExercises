import numpy as np
import matplotlib.pyplot as plt

def fitness(chromosome: np.ndarray):
    assert np.all((chromosome == 0) | (chromosome == 1)), 'Chromosome is not binary'
    powers = 2 ** np.arange(len(chromosome)-1, -1, -1)
    return np.sum(chromosome * powers)

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

def run_genetic_algorithm(P_delta: float, P_mu=0.001, n=100, l=100, max_generations=100):
    population = np.random.randint(0, 2, size=(n, l))
    best_fitness_per_generation = []
    mean_fitness_per_generation = []

    for generation in range(max_generations):
        fitnesses = np.array([fitness(chromosome) for chromosome in population])
        best_fitness_per_generation.append(np.max(fitnesses))
        mean_fitness_per_generation.append(np.mean(fitnesses))

        new_population = []
        while len(new_population) < n:
            parent1, parent2 = roulette_wheel_selection(population, fitnesses)
            child1, child2 = one_point_crossover(parent1, parent2, P_delta)
            new_population.append(mutate(child1, P_mu))
            new_population.append(mutate(child2, P_mu))
        population = np.array(new_population[:n])

    return best_fitness_per_generation, mean_fitness_per_generation

def main():
    P_delta = 0.7
    bests, means = run_genetic_algorithm(P_delta)

    generations = np.arange(len(bests))


if __name__ == "__main__":
    main()
