from fuzzy.operations import plot_fuzzy, fuzzy_mult, fuzzy_div, fuzzy_sub

# Define fuzzy numbers A and B as dictionaries
A = {6: 0.33, 7: 0.67, 8: 1.00, 9: 0.67, 10: 0.33}
B = {1: 0.33, 2: 0.67, 3: 1.00, 4: 0.67, 5: 0.33}

# (i) Plot A and B
plot_fuzzy(A, "Fuzzy Number A", '../results/3_1_a.png')
plot_fuzzy(B, "Fuzzy Number B", '../results/3_1_b.png')

# (ii) Fuzzy multiplication using extension principle (max-min convolution)
plot_fuzzy(fuzzy_mult(A, B), title="Fuzzy Product C = A * B", filename='../results/3_1_mult.png')

# (iii) Fuzzy subtraction and division using similar principle
plot_fuzzy(fuzzy_sub(A, B), title="Fuzzy Difference D = A - B", filename='../results/3_1_sub.png')
plot_fuzzy(fuzzy_div(A, B), title="Fuzzy Division E = A / B", filename='../results/3_1_div.png')
