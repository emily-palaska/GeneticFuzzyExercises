from genetic.genetic_algorithm import GeneticAlgorithm, linear_fitness, binary_fitness, schema
from genetic.plot_functions import schema_count

def main():
    schemas = [
        '1' + '*'*99,
        '*'*99 + '1',
        '111' + '*'*97,
        '*1*1*1' + '*'*94,
        '*'*100,
        '*1'*50,
        '0'*100,
        '1'*100,
        '0' + '*'*98 + '0',
        '1' + '*'*98 + '1'
    ]
    labels = [
        '1***...',
        '...***1',
        '111***...',
        '*1*1*1***...',
        '***...',
        '*1*1*1...',
        '000...',
        '111...',
        '0***...***0',
        '1***...***1'
    ]
    genetic = GeneticAlgorithm(target=schema, fitness=linear_fitness)
    counts = genetic.run(schemas=schemas)
    schema_count(counts, labels, '../results/9_3_counts_linear.png')


if __name__ == '__main__':
    main()