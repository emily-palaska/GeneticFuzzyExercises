from abstraction import GeneticAlgorithm, binary_fitness, best
from plotting import generation_progress, multi_generation_progress

def main():
    genetic = GeneticAlgorithm(fitness=binary_fitness, target=best)

    best_fitness, mean_fitness = genetic.run(generations=100)
    generation_progress(best_fitness, mean_fitness, path='../results/9_2_progress.png')

    genetic = GeneticAlgorithm(fitness=binary_fitness, target=best)
    bests, means, labels = [], [], []
    for i in range(4):
        n=10**i
        print(n)
        genetic.n=n
        best_fitness, mean_fitness = genetic.run(generations=100)
        bests.append(best_fitness)
        means.append(mean_fitness)
        labels.append(f'n={n}')
    multi_generation_progress(bests, means, labels, '../results/9_2_multi_n.png')

    genetic = GeneticAlgorithm(fitness=binary_fitness, target=best)
    bests, means, labels = [], [], []
    for i in range(6):
        p_delta = 0.2 * i
        print(p_delta)
        genetic.p_delta=p_delta
        best_fitness, mean_fitness = genetic.run(generations=100)
        bests.append(best_fitness)
        means.append(mean_fitness)
        labels.append(f'p_delta={p_delta:.1f}')
    multi_generation_progress(bests, means, labels, '../results/9_2_multi_p_delta.png')

    genetic = GeneticAlgorithm(fitness=binary_fitness, target=best)
    p_mus = [0.0, 0.001, 0.05, 0.1, 0.2, 0.3]
    bests, means, labels = [], [], []
    for p_mu in p_mus:
        print(p_mu)
        genetic.p_mu=p_mu
        best_fitness, mean_fitness = genetic.run(generations=100)
        bests.append(best_fitness)
        means.append(mean_fitness)
        labels.append(f'p_mu={p_mu:.3f}')
    multi_generation_progress(bests, means, labels, '../results/9_2_multi_p_mu.png')


if __name__ == "__main__":
    main()
