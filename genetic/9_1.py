import numpy as np
from genetic_algorithm import GeneticAlgorithm, linear_fitness
from plot_functions import p_delta_histograms

def main():
    runs, p_deltas = 20, [0.7, 0.0]
    P_mu, n, l = 0.001, 100, 100
    results = []

    for P_delta in p_deltas:
        ga = GeneticAlgorithm(P_delta=P_delta, fitness=linear_fitness, P_mu=P_mu, n=n, l=l)
        generations = []
        for r in range(runs):
            generations.append(ga.run())
            print(f'\tRun {r}: {generations[-1]}')
        results.append(generations)
        print(f"Average generation for P_delta = {P_delta}: {np.mean(generations):.2f}")
    p_delta_histograms(results, p_deltas, '../results/9_1_hist.png')

if __name__ == "__main__":
    main()
