import torch
from genetic.utils import mutate, roulette_wheel, one_point_crossover

def linear_fitness(population: torch.Tensor):
    return population.sum(dim=1)

def binary_fitness(population: torch.Tensor):
    batch_size, l = population.shape
    device = population.device

    power_range = torch.arange(l - 1, -1, -1, device=device, dtype=torch.float64)
    fitness = (population * 2 ** power_range).sum(dim=1)
    return fitness

def ones(fitnesses: torch.Tensor, target=None):
    return torch.any(fitnesses == target).item()

def best(fitnesses: torch.Tensor):
    return torch.argmax(fitnesses).item()

def schema(population: torch.Tensor, schemas: list):
    batch_size, chrom_len = population.shape
    counts, device = [], population.device

    for s in schemas:
        assert len(s) == chrom_len, f"Schema length must match chromosome length: \n{s}"

        mask = torch.tensor([c != '*' for c in s], dtype=torch.bool, device=device)
        target = torch.tensor(
            [int(c) if c != '*' else 0 for c in s], dtype=torch.uint8, device=device
        )
        selected, expected = population[:, mask], target[mask]
        matches = (selected == expected).all(dim=1)
        counts.append(matches.sum().item())

    return counts

class GeneticAlgorithm:
    def __init__(self, fitness=linear_fitness, selection=roulette_wheel, target=ones,
                 P_delta=0.7, P_mu=0.001, n=100, l=100):
        self.P_delta, self.P_mu = P_delta, P_mu
        self.n, self.l = n, l
        self.fitness = fitness
        self.selection = selection
        self.target = target
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self, generations=100, schemas=None):
        population = torch.randint(0, 2, (self.n, self.l), dtype=torch.int32, device=self.device)
        best_fitness, mean_fitness, counts = [], [], []

        for generation in range(int(generations)):
            fitnesses = self.fitness(population)

            if self.target==ones and self.target(fitnesses, self.l):
                return generation
            elif self.target==best:
                ind = self.target(fitnesses)
                best_fitness.append(fitnesses[ind].cpu())
                mean_fitness.append(fitnesses.mean().cpu().item())
            elif self.target == schema:
                counts.append(self.target(population, schemas))

            probabilities = fitnesses / torch.sum(fitnesses)
            parent_indices = torch.multinomial(probabilities, num_samples=2 * self.n, replacement=True)
            parents = population[parent_indices].reshape(self.n, 2, -1)
            children = one_point_crossover(parents, self.P_delta)
            mutated_children = mutate(children, self.P_mu)

            population =  mutated_children[:self.n]

        if self.target==ones: return generations
        if self.target==best: return best_fitness, mean_fitness
        if self.target==schema: return counts

