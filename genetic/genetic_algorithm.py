import torch
def linear_fitness(population: torch.Tensor):
    return population.sum(dim=1)

def binary_fitness(population: torch.Tensor):
    batch_size, chromosome_length = population.shape

    # Use int64 if chromosome length is safe
    if chromosome_length <= 63:
        powers = 2 ** torch.arange(chromosome_length - 1, -1, -1, device=population.device, dtype=torch.int64)
        fitness = (population.to(torch.int64) * powers).sum(dim=1)
        return fitness.to(torch.float32)

    # For long chromosomes: process in chunks to avoid overflow
    chunk_size = 32
    fitness = torch.zeros(batch_size, dtype=torch.float64, device=population.device)

    for i in range(0, chromosome_length, chunk_size):
        end = min(i + chunk_size, chromosome_length)
        chunk = population[:, i:end].to(torch.float64)
        power_range = torch.arange(end - 1, i - 1, -1, device=population.device, dtype=torch.float64)
        powers = 2 ** power_range
        fitness += (chunk * powers).sum(dim=1)

    return fitness.to(torch.float32)

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

def ones(fitnesses: torch.Tensor, target=None):
    return torch.any(fitnesses == target).item()

def best(fitnesses: torch.Tensor, target=None):
    return torch.argmax(fitnesses).item()

class GeneticAlgorithm:
    def __init__(self, fitness=linear_fitness, selection=roulette_wheel, target=ones,
                 P_delta=0.7, P_mu=0.001, n=100, l=100):
        self.P_delta, self.P_mu = P_delta, P_mu
        self.n, self.l = n, l
        self.fitness = fitness
        self.selection = selection
        self.target = target
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self, generations=1e3):
        population = torch.randint(0, 2, (self.n, self.l), dtype=torch.int32, device=self.device)
        best_fitness, mean_fitness = [], []

        for generation in range(int(generations)):
            fitnesses = self.fitness(population)

            if self.target==ones and self.target(fitnesses, self.l):
                return generation
            elif self.target==best:
                ind = self.target(fitnesses)

                best_fitness.append(fitnesses[ind].cpu())
                mean_fitness.append(fitnesses.mean().cpu().item())

            probabilities = fitnesses / torch.sum(fitnesses)
            parent_indices = torch.multinomial(probabilities, num_samples=2 * self.n, replacement=True)
            parents = population[parent_indices].reshape(self.n, 2, -1)
            children = one_point_crossover(parents, self.P_delta)
            mutated_children = mutate(children, self.P_mu)

            population =  mutated_children[:self.n]

        if self.target==ones: return generations
        if self.target==best: return best_fitness, mean_fitness

