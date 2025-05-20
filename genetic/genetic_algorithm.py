from utils import *

class GeneticAlgorithm:
    def __init__(self, fitness=linear_fitness, selection=roulette_wheel_selection,
                 P_delta=0.7, P_mu=0.001, n=100, l=100, library='torch'):
        self.P_delta, self.P_mu = P_delta, P_mu
        self.n, self.l = n, l
        self.fitness = fitness
        self.selection = selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.library = library

    def run(self, max_generations=1e6):
        if self.library == 'numpy': population = np.random.randint(0, 2, size=(self.n, self.l))
        else: population = torch.randint(0, 2, (self.n, self.l), dtype=torch.int32, device=self.device)

        for generation in range(int(max_generations)):
            fitnesses = self.fitness(population)

            if target_check(fitnesses, self.l):
                return generation

            population = self.evolve(population, fitnesses)

        return max_generations

    def evolve(self, population, fitnesses):
        if self.library == 'numpy':
            probabilities = fitnesses / np.sum(fitnesses)
            parent_indices = np.random.choice(len(population), size=2 * self.n, p=probabilities)
            parents = population[parent_indices].reshape(self.n, 2, -1)

            children = []
            for parent_pair in parents:
                child1, child2 = one_point_crossover(parent_pair[0], parent_pair[1], self.P_delta)
                children.append(mutate(child1, self.P_mu))
                children.append(mutate(child2, self.P_mu))

            return np.array(children[:self.n])

        elif self.library == 'torch':
            probabilities = fitnesses / torch.sum(fitnesses)
            parent_indices = torch.multinomial(probabilities, num_samples=2 * self.n, replacement=True)
            parents = population[parent_indices].reshape(self.n, 2, -1)

            children = []
            for parent_pair in parents:
                child1, child2 = one_point_crossover(parent_pair[0], parent_pair[1], self.P_delta)
                children.append(mutate(child1, self.P_mu))
                children.append(mutate(child2, self.P_mu))

            return torch.stack(children[:self.n])


