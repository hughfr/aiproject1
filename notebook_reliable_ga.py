#!/usr/bin/env python3
"""
Reliable GA for Notebook - Simple version to add to your notebook
"""

import random
import time

def solve_ga_reliable(sud, pop_size=200, crossover_rate=0.77, mutation_rate=0.043, 
                     tournament_k=5, max_generations=1000, max_attempts=10):
    """
    Reliable GA solver that tries multiple seeds until success
    
    Args:
        sud: Sudoku object
        pop_size: Population size
        crossover_rate: Crossover rate
        mutation_rate: Mutation rate  
        tournament_k: Tournament size
        max_generations: Max generations per attempt
        max_attempts: Max number of attempts with different seeds
    
    Returns:
        (solution, total_runtime, total_generations, attempts_used)
    """
    print(f"🔄 Reliable GA Solver - trying up to {max_attempts} attempts")
    
    total_start_time = time.time()
    total_generations = 0
    
    for attempt in range(max_attempts):
        # Use different seed for each attempt
        seed = random.randint(1, 10000)
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"  Attempt {attempt + 1}/{max_attempts}: seed {seed}")
        
        # Single GA run
        solution, generations, success = solve_ga_single_attempt(
            sud, pop_size, crossover_rate, mutation_rate, 
            tournament_k, max_generations
        )
        
        total_generations += generations
        
        if success:
            total_runtime = time.time() - total_start_time
            print(f"  ✅ SUCCESS! Runtime: {total_runtime:.1f}s, Generations: {total_generations}")
            return solution, total_runtime, total_generations, attempt + 1
        else:
            conflicts = total_conflicts(solution)
            print(f"  ❌ Failed. Conflicts: {conflicts}, Generations: {generations}")
    
    total_runtime = time.time() - total_start_time
    print(f"  ❌ Failed after {max_attempts} attempts, {total_runtime:.1f}s total")
    return None, total_runtime, total_generations, max_attempts

def solve_ga_single_attempt(sud, pop_size, crossover_rate, mutation_rate, 
                           tournament_k, max_generations):
    """Single GA attempt with current random seed"""
    # Initialize population
    population = [random_complete_assignment(sud) for _ in range(pop_size)]
    
    for gen in range(max_generations):
        # Check if any individual is solved
        for individual in population:
            if total_conflicts(individual) == 0:
                return individual, gen, True
        
        # Calculate fitnesses
        fitnesses = [ga_fitness(x) for x in population]
        
        # Progress reporting every 200 generations
        if gen % 200 == 0 and gen > 0:
            best_fitness = max(fitnesses)
            best_conflicts = -best_fitness
            print(f"    Generation {gen}: best_conflicts={best_conflicts}")
        
        # Create offspring
        offspring = []
        
        while len(offspring) < pop_size:
            # Selection
            p1 = ga_tournament_select(population, fitnesses, tournament_k)
            p2 = ga_tournament_select(population, fitnesses, tournament_k)
            
            # Crossover
            if random.random() < crossover_rate:
                c1, c2 = ga_crossover(p1, p2)
            else:
                c1 = {'grid': p1['grid'].copy(), 'givens_mask': p1['givens_mask']}
                c2 = {'grid': p2['grid'].copy(), 'givens_mask': p2['givens_mask']}
            
            # Mutation
            c1 = ga_mutate(c1, mutation_rate, sud)
            c2 = ga_mutate(c2, mutation_rate, sud)
            
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)
        
        # Replace population
        population = offspring
    
    # Return best found
    fitnesses = [ga_fitness(x) for x in population]
    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    return population[best_idx], max_generations, False

# Example usage for your notebook:
"""
# Replace your current solve_ga call with:
solution, runtime, generations, attempts = solve_ga_reliable(
    sud, 
    pop_size=200, 
    crossover_rate=0.77, 
    mutation_rate=0.043, 
    tournament_k=5, 
    max_generations=1000, 
    max_attempts=10
)

if solution is not None:
    print(f"✅ GA SOLVED in {runtime:.1f}s after {attempts} attempts!")
    print(f"Total generations: {generations}")
else:
    print(f"❌ GA failed after {attempts} attempts")
"""
