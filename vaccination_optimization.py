"""
NSGA-II implementation for optimizing vaccination strategies in epidemic networks.
This implementation aims to find Pareto-optimal solutions that balance between:
1. Minimizing the peak infection size
2. Minimizing the number of vaccinations needed
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import csv

from epidemic_simulation import run_simulation

# Create Barabási-Albert network for modeling social contacts
num_nodes = 400
G = nx.barabasi_albert_graph(num_nodes, 3)
# G = nx.erdos_renyi_graph(num_nodes, 0.01)
# G = nx.watts_strogatz_graph(num_nodes, 4, 0.1)

dim = G.number_of_nodes()

def evaluate(individual):
    """
    Evaluate vaccination strategy against two objectives:
    f1: Peak infection size (lower is better)
    f2: Number of vaccinations (lower is more economical)
    """
    x_vector = individual['x']
    f1 = run_simulation(x_vector, G, repeats=8)
    f2 = sum(x_vector)
    return np.array([f1, f2])

def bit_flip_mutation(individual, mutation_rate=0.05):
    """Mutation operator that randomly flips vaccination status (0->1 or 1->0)"""
    mutant = individual.copy()
    mutant['x'] = np.array([
        1 - bit if random.random() < mutation_rate else bit
        for bit in individual['x']
    ])
    mutant['objectives'] = evaluate(mutant)
    return mutant

def tournament_selection(population):
    """Select best individual from two random candidates based on rank and crowding distance"""
    i, j = random.sample(range(len(population)), 2)
    p1, p2 = population[i], population[j]
    if p1['rank'] < p2['rank']:
        return p1
    elif p1['rank'] > p2['rank']:
        return p2
    else:
        return p1 if p1['distance'] > p2['distance'] else p2

def dominates(a, b):
    """Check if solution a dominates solution b in objective space"""
    return np.all(a <= b) and np.any(a < b)

# Non-dominated sorting: returns a list of fronts (each front is a list of indices).
def non_dominated_sort(population):
    S = [[] for _ in range(len(population))]
    n = [0 for _ in range(len(population))]
    rank = [0 for _ in range(len(population))]
    fronts = [[]]
    
    for p in range(len(population)):
        S[p] = []
        n[p] = 0
        for q in range(len(population)):
            if dominates(population[p]['objectives'], population[q]['objectives']):
                S[p].append(q)
            elif dominates(population[q]['objectives'], population[p]['objectives']):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 1
            fronts[0].append(p)
    
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    if not fronts[-1]:
        fronts.pop()
    
    for i, front in enumerate(fronts):
        for idx in front:
            population[idx]['rank'] = i + 1
    return fronts

# Crowding distance assignment for individuals in a front.
def crowding_distance(population, front):
    l = len(front)
    if l == 0:
        return
    for idx in front:
        population[idx]['distance'] = 0
    
    num_objectives = len(population[front[0]]['objectives'])
    
    for m in range(num_objectives):
        front_objs = [population[idx]['objectives'][m] for idx in front]
        sorted_idx = np.argsort(front_objs)
        f_max = front_objs[sorted_idx[-1]]
        f_min = front_objs[sorted_idx[0]]
        
        # Assign infinite distance to boundary points.
        population[front[sorted_idx[0]]]['distance'] = float('inf')
        population[front[sorted_idx[-1]]]['distance'] = float('inf')
        
        if f_max == f_min:
            continue
        
        for i in range(1, l - 1):
            next_obj = population[front[sorted_idx[i + 1]]]['objectives'][m]
            prev_obj = population[front[sorted_idx[i - 1]]]['objectives'][m]
            population[front[sorted_idx[i]]]['distance'] += (next_obj - prev_obj) / (f_max - f_min)

def create_offspring(population, offspring_size, sigma=0.1, lower_bound=-10, upper_bound=10):
    """Create new generation using tournament selection and bit-flip mutation"""
    offspring = []
    while len(offspring) < offspring_size:
        parent = tournament_selection(population)
        child = bit_flip_mutation(parent)
        offspring.append(child)
    return offspring

def nsga2(pop_size=20, generations=10, sigma=0.1, lower_bound=-10, upper_bound=10):
    """
    Main NSGA-II algorithm implementation
    
    Args:
        pop_size: Size of population in each generation
        generations: Number of generations to evolve
        sigma: Parameter for mutation (not used in binary encoding)
        lower/upper_bound: Parameters for continuous optimization (not used here)
    """
    # Initialize population randomly
    population = []
    for _ in range(pop_size):
        ind = {}
        ind['x'] = np.random.choice([0, 1], size=dim)
        ind['objectives'] = evaluate(ind)
        ind['rank'] = None
        ind['distance'] = None
        population.append(ind)
    
    # Initial non-dominated sorting
    fronts = non_dominated_sort(population)
    for front in fronts:
        crowding_distance(population, front)
    
    initial_population = [ind.copy() for ind in population]
    
    # Main evolution loop
    for gen in range(generations):
        # Create offspring and combine with parents
        offspring = create_offspring(population, pop_size, sigma, lower_bound, upper_bound)
        combined_population = population + offspring
        
        # Non-dominated sorting of combined population
        fronts = non_dominated_sort(combined_population)
        for front in fronts:
            crowding_distance(combined_population, front)
        
        # Select next generation
        new_population = []
        i = 0
        while len(new_population) + len(fronts[i]) <= pop_size:
            for idx in fronts[i]:
                new_population.append(combined_population[idx])
            i += 1
        
        # Fill remaining slots with best individuals from current front
        remaining = pop_size - len(new_population)
        if remaining > 0:
            last_front = [combined_population[idx] for idx in fronts[i]]
            last_front.sort(key=lambda ind: ind['distance'], reverse=True)
            new_population.extend(last_front[:remaining])
        
        population = new_population
        best_front = [combined_population[idx] for idx in fronts[0]]
        print(f"Generation {gen + 1}: Best front size = {len(best_front)}")
    
    return initial_population, population

if __name__ == "__main__":
    # Run NSGA-II optimization
    init_pop, final_pop = nsga2(pop_size=50, generations=20, sigma=0.1)

    # Plot population development
    init_objs = np.array([ind['objectives'] for ind in init_pop])
    final_objs = np.array([ind['objectives'] for ind in final_pop])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(init_objs[:, 1], init_objs[:, 0], color='red', label='Initial Population')
    plt.scatter(final_objs[:, 1], final_objs[:, 0], color='blue', label='Final Population')
    plt.xlabel('f2(x): Number of vaccinated individuals')
    plt.ylabel('f1(x): Peak infection (lower is better)')
    plt.title('Objective Space: Initial (red) and Final (blue) Populations')
    plt.legend()
    plt.grid(True)
    plt.savefig("NSGA2_scatter.png", dpi=300)
    plt.close()
    
    # Extract and analyze Pareto front
    pareto_pop = [ind for ind in final_pop if ind['rank'] == 1]
    pareto_objs = np.array([ind['objectives'] for ind in pareto_pop])
    
    # Save Pareto-optimal solutions
    with open("pareto_strategies.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_vector", "f1_peak", "f2_vaccinated"])
        for ind in pareto_pop:
            writer.writerow([list(ind['x']), ind['objectives'][0], ind['objectives'][1]])
    
    # Plot Pareto front
    if len(pareto_objs) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(pareto_objs[:, 1], pareto_objs[:, 0], color='blue', label='Pareto Front')
        plt.xlabel('Number of Vaccinated Nodes (f2)')
        plt.ylabel('Peak Infection Size (f1)')
        plt.title('Pareto Front of Vaccination Strategies')
        plt.legend()
        plt.grid(True)
        plt.savefig("NSGA2_pareto_only.png", dpi=300)
        plt.close()
    
    # Analyze solutions
    print("\nNetwork statistics:")
    all_degrees = [G.degree(node) for node in G.nodes()]
    print(f"Average degree in network: {np.mean(all_degrees):.2f}")
    print(f"Minimum degree: {min(all_degrees)}")
    print(f"Maximum degree: {max(all_degrees)}")
    
    print("\nAnalysis of Pareto-optimal solutions:")
    for i, solution in enumerate(pareto_pop):
        x_vector = solution['x']
        vaccinated_nodes = np.where(x_vector == 1)[0]
        f1, f2 = solution['objectives']
        print(f"\nSolution {i+1}:")
        print(f"Peak infection: {f1:.2f}")
        print(f"Number of vaccinated: {f2}")
        degrees = [G.degree(node) for node in vaccinated_nodes]
        print(f"Average degree of vaccinated nodes: {np.mean(degrees):.2f}")
