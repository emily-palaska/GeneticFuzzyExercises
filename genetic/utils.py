import torch

def mutate(population: torch.Tensor, P_mu: float):
    mutation_mask = torch.rand_like(population, dtype=torch.float32) < P_mu
    return population ^ mutation_mask.to(torch.bool)

def roulette_wheel(population: torch.Tensor, fitnesses: torch.Tensor):
    probabilities = fitnesses / torch.sum(fitnesses)
    indices = torch.multinomial(probabilities, num_samples=2, replacement=True)
    return population[indices[0]], population[indices[1]]

def one_point_crossover(parents: torch.Tensor, P_delta: float):
    n, _, gene_length = parents.shape
    do_crossover = torch.rand(n) < P_delta
    points = torch.randint(1, gene_length, (n,))
    child1, child2 = parents[:, 0, :].clone(), parents[:, 1, :].clone()

    for i in range(n):
        if do_crossover[i]:
            p = points[i]
            child1[i, p:], child2[i, p:] = parents[i, 1, p:], parents[i, 0, p:]

    return torch.cat([child1, child2], dim=0)