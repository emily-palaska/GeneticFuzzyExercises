import numpy as np
import torch

def linear_fitness(population):
    if isinstance(population, np.ndarray):
        return np.sum(population, axis=1)
    if isinstance(population, torch.Tensor):
        return population.sum(dim=1)

def binary_fitness(chromosome):
    if isinstance(chromosome, np.ndarray):
        powers = 2 ** np.arange(len(chromosome) - 1, -1, -1)
        return np.sum(chromosome * powers)
    if isinstance(chromosome, torch.Tensor):
        powers = 2 ** torch.arange(len(chromosome) - 1, -1, -1, device=chromosome.device)
        return (chromosome * powers).sum()

def mutate(chromosome, P_mu: float):
    if isinstance(chromosome, np.ndarray):
        mutation_mask = np.random.rand(len(chromosome)) < P_mu
        chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
        return chromosome
    if isinstance(chromosome, torch.Tensor):
        mutation_mask = torch.rand_like(chromosome, dtype=torch.float32) < P_mu
        return chromosome ^ mutation_mask.to(torch.bool)


def roulette_wheel_selection(population, fitnesses):
    if isinstance(population, np.ndarray):
        probabilities = fitnesses / np.sum(fitnesses)
        indices = np.random.choice(len(population), size=2, p=probabilities)
        return population[indices[0]], population[indices[1]]
    if isinstance(population, torch.Tensor):
        probabilities = fitnesses / torch.sum(fitnesses)
        indices = torch.multinomial(probabilities, num_samples=2, replacement=True)
        return population[indices[0]], population[indices[1]]

def one_point_crossover(parent1, parent2, P_delta: float):
    if isinstance(parent1, np.ndarray):
        if np.random.rand() < P_delta:
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return np.copy(parent1), np.copy(parent2)
    if isinstance(parent1, torch.Tensor):
        if torch.rand(1).item() < P_delta:
            point = torch.randint(1, parent1.size(0), (1,)).item()
            child1 = torch.cat((parent1[:point], parent2[point:]))
            child2 = torch.cat((parent2[:point], parent1[point:]))
            return child1.clone(), child2.clone()
        return parent1.clone(), parent2.clone()

def target_check(fitnesses, target):
    if isinstance(fitnesses, np.ndarray):
        if np.any(fitnesses == target): return True
    if isinstance(fitnesses, torch.Tensor):
        if torch.any(fitnesses, target): return True
    return False