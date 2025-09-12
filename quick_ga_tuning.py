#!/usr/bin/env python3
"""
Quick GA Hyperparameter Tuning Script

A simplified version for fast hyperparameter tuning with minimal dependencies.
Focuses on the most important parameters with quick evaluation.
"""

import numpy as np
import time
import random
import json
import os
from typing import Dict, List, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your GA implementation
from sudoku_solver2 import (
    read_puzzle_txt, sudoku_from_grid, solve_ga, 
    total_conflicts, is_solved, reset_decisions, decisions
)

class QuickGATuner:
    """Simplified GA hyperparameter tuner"""
    
    def __init__(self, puzzle_path: str, max_workers: int = 4):
        self.puzzle_path = puzzle_path
        self.max_workers = max_workers
        self.results = []
        
    def evaluate_config(self, pop_size: int, crossover_rate: float, 
                       mutation_rate: float, tournament_k: int, 
                       max_generations: int = 500, timeout: int = 120) -> Dict:
        """Evaluate a single parameter configuration"""
        start_time = time.time()
        
        try:
            # Read puzzle
            grid = read_puzzle_txt(self.puzzle_path)
            sud = sudoku_from_grid(grid)
            
            # Reset decision counter
            reset_decisions()
            
            # Run GA with early stopping
            solution = self._run_ga_quick(sud, pop_size, crossover_rate, 
                                       mutation_rate, tournament_k, 
                                       max_generations, timeout)
            
            runtime = time.time() - start_time
            
            # Analyze results
            if solution is not None:
                success = is_solved(solution)
                final_conflicts = total_conflicts(solution)
            else:
                success = False
                final_conflicts = float('inf')
            
            return {
                'pop_size': pop_size,
                'crossover_rate': crossover_rate,
                'mutation_rate': mutation_rate,
                'tournament_k': tournament_k,
                'success': success,
                'final_conflicts': final_conflicts,
                'decisions': decisions,
                'runtime': runtime
            }
            
        except Exception as e:
            return {
                'pop_size': pop_size,
                'crossover_rate': crossover_rate,
                'mutation_rate': mutation_rate,
                'tournament_k': tournament_k,
                'success': False,
                'final_conflicts': float('inf'),
                'decisions': 0,
                'runtime': time.time() - start_time,
                'error': str(e)
            }
    
    def _run_ga_quick(self, sud, pop_size, crossover_rate, mutation_rate, 
                     tournament_k, max_generations, timeout):
        """Quick GA implementation with early stopping"""
        start_time = time.time()
        
        # Initialize population
        population = [self._random_complete_assignment(sud) for _ in range(pop_size)]
        
        for gen in range(max_generations):
            # Check timeout
            if time.time() - start_time > timeout:
                break
                
            # Check if any individual is solved
            for individual in population:
                if total_conflicts(individual) == 0:
                    return individual
            
            # Calculate fitnesses
            fitnesses = [self._ga_fitness(x) for x in population]
            
            # Create offspring
            offspring = []
            
            while len(offspring) < pop_size:
                # Selection
                p1 = self._ga_tournament_select(population, fitnesses, tournament_k)
                p2 = self._ga_tournament_select(population, fitnesses, tournament_k)
                
                # Crossover
                if random.random() < crossover_rate:
                    c1, c2 = self._ga_crossover(p1, p2)
                else:
                    c1 = {'grid': p1['grid'].copy(), 'givens_mask': p1['givens_mask']}
                    c2 = {'grid': p2['grid'].copy(), 'givens_mask': p2['givens_mask']}
                
                # Mutation
                c1 = self._ga_mutate(c1, mutation_rate, sud)
                c2 = self._ga_mutate(c2, mutation_rate, sud)
                
                offspring.append(c1)
                if len(offspring) < pop_size:
                    offspring.append(c2)
            
            # Replace population
            population = offspring
        
        # Return best found
        fitnesses = [self._ga_fitness(x) for x in population]
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        return population[best_idx]
    
    def _random_complete_assignment(self, sud):
        """Create random complete assignment respecting givens"""
        C = sud['grid'].copy()
        givens_mask = sud['givens_mask']
        
        for br in range(3):
            for bc in range(3):
                given_values = set()
                cells = []
                for r in range(br*3, (br+1)*3):
                    for c in range(bc*3, (bc+1)*3):
                        if givens_mask[r, c]:
                            given_values.add(C[r, c])
                        else:
                            cells.append((r, c))
                
                missing = set(range(1, 10)) - given_values
                missing_list = list(missing)
                random.shuffle(missing_list)
                
                for i, (r, c) in enumerate(cells):
                    if i < len(missing_list):
                        C[r, c] = missing_list[i]
        
        return {'grid': C, 'givens_mask': givens_mask}
    
    def _ga_fitness(self, individual):
        """Fitness function: higher is better"""
        return -total_conflicts(individual)
    
    def _ga_tournament_select(self, population, fitnesses, k):
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), k)
        best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
        return {'grid': population[best_idx]['grid'].copy(), 'givens_mask': population[best_idx]['givens_mask']}
    
    def _ga_crossover(self, p1, p2):
        """Block-based crossover"""
        child1 = {'grid': p1['grid'].copy(), 'givens_mask': p1['givens_mask']}
        child2 = {'grid': p2['grid'].copy(), 'givens_mask': p2['givens_mask']}
        
        for br in range(3):
            for bc in range(3):
                if random.random() < 0.5:
                    for r in range(br*3, (br+1)*3):
                        for c in range(bc*3, (bc+1)*3):
                            child1['grid'][r, c] = p1['grid'][r, c]
                            child2['grid'][r, c] = p2['grid'][r, c]
                else:
                    for r in range(br*3, (br+1)*3):
                        for c in range(bc*3, (bc+1)*3):
                            child1['grid'][r, c] = p2['grid'][r, c]
                            child2['grid'][r, c] = p1['grid'][r, c]
        
        # Repair givens
        givens_mask = p1['givens_mask']
        for r in range(9):
            for c in range(9):
                if givens_mask[r, c]:
                    child1['grid'][r, c] = p1['grid'][r, c]
                    child2['grid'][r, c] = p1['grid'][r, c]
        
        return child1, child2
    
    def _ga_mutate(self, individual, rate, sud):
        """Mutate using swap strategies"""
        mutated = {'grid': individual['grid'].copy(), 'givens_mask': individual['givens_mask']}
        
        if random.random() < rate:
            if random.random() < 0.7:
                # Swap within same box
                br, bc = random.randint(0, 2), random.randint(0, 2)
                cells = []
                for r in range(br*3, (br+1)*3):
                    for c in range(bc*3, (bc+1)*3):
                        if not sud['givens_mask'][r, c]:
                            cells.append((r, c))
                
                if len(cells) >= 2:
                    cell1, cell2 = random.sample(cells, 2)
                    r1, c1 = cell1
                    r2, c2 = cell2
                    mutated['grid'][r1, c1], mutated['grid'][r2, c2] = mutated['grid'][r2, c2], mutated['grid'][r1, c1]
            else:
                # Swap across boxes
                non_given_cells = []
                for r in range(9):
                    for c in range(9):
                        if not sud['givens_mask'][r, c]:
                            non_given_cells.append((r, c))
                
                if len(non_given_cells) >= 2:
                    cell1, cell2 = random.sample(non_given_cells, 2)
                    r1, c1 = cell1
                    r2, c2 = cell2
                    
                    box1 = (r1 // 3, c1 // 3)
                    box2 = (r2 // 3, c2 // 3)
                    
                    if box1 != box2:
                        mutated['grid'][r1, c1], mutated['grid'][r2, c2] = mutated['grid'][r2, c2], mutated['grid'][r1, c1]
        
        return mutated
    
    def quick_grid_search(self, timeout: int = 60):
        """Quick grid search with reduced parameter space"""
        print("Starting quick grid search...")
        
        # Reduced parameter space for quick testing
        param_combinations = [
            # (pop_size, crossover_rate, mutation_rate, tournament_k)
            (100, 0.8, 0.03, 5),   # Current settings
            (150, 0.8, 0.03, 5),   # Larger population
            (100, 0.9, 0.03, 5),   # Higher crossover
            (100, 0.8, 0.05, 5),   # Higher mutation
            (100, 0.8, 0.03, 7),   # Larger tournament
            (200, 0.9, 0.05, 7),   # Combined improvements
            (50, 0.7, 0.01, 3),    # Conservative settings
            (300, 0.95, 0.1, 10),  # Aggressive settings
        ]
        
        print(f"Testing {len(param_combinations)} configurations...")
        
        results = []
        for i, (pop_size, crossover_rate, mutation_rate, tournament_k) in enumerate(param_combinations):
            print(f"Testing config {i+1}/{len(param_combinations)}: "
                  f"pop={pop_size}, cross={crossover_rate}, mut={mutation_rate}, k={tournament_k}")
            
            result = self.evaluate_config(pop_size, crossover_rate, mutation_rate, 
                                        tournament_k, max_generations=300, timeout=timeout)
            results.append(result)
            
            # Print immediate results
            if result['success']:
                print(f"  ✓ SUCCESS! Runtime: {result['runtime']:.1f}s, Decisions: {result['decisions']}")
            else:
                print(f"  ✗ Failed. Conflicts: {result['final_conflicts']}, Runtime: {result['runtime']:.1f}s")
        
        self.results = results
        return results
    
    def random_search(self, n_trials: int = 20, timeout: int = 60):
        """Random search over parameter space"""
        print(f"Starting random search with {n_trials} trials...")
        
        results = []
        for i in range(n_trials):
            # Random parameters
            pop_size = random.choice([50, 100, 150, 200, 300])
            crossover_rate = random.uniform(0.6, 0.95)
            mutation_rate = random.uniform(0.01, 0.1)
            tournament_k = random.choice([3, 5, 7, 10])
            
            print(f"Trial {i+1}/{n_trials}: "
                  f"pop={pop_size}, cross={crossover_rate:.2f}, mut={mutation_rate:.3f}, k={tournament_k}")
            
            result = self.evaluate_config(pop_size, crossover_rate, mutation_rate, 
                                        tournament_k, max_generations=300, timeout=timeout)
            results.append(result)
            
            # Print immediate results
            if result['success']:
                print(f"  ✓ SUCCESS! Runtime: {result['runtime']:.1f}s, Decisions: {result['decisions']}")
            else:
                print(f"  ✗ Failed. Conflicts: {result['final_conflicts']}, Runtime: {result['runtime']:.1f}s")
        
        self.results = results
        return results
    
    def analyze_results(self):
        """Analyze and print results"""
        if not self.results:
            print("No results to analyze.")
            return
        
        print("\n" + "="*60)
        print("QUICK TUNING RESULTS")
        print("="*60)
        
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Successful solves: {len(successful)}")
        print(f"Success rate: {len(successful)/len(self.results)*100:.1f}%")
        
        if successful:
            print(f"\nSuccessful configurations:")
            # Sort by runtime (faster is better)
            successful.sort(key=lambda x: x['runtime'])
            
            for i, result in enumerate(successful[:5], 1):
                print(f"  {i}. pop={result['pop_size']}, cross={result['crossover_rate']:.2f}, "
                      f"mut={result['mutation_rate']:.3f}, k={result['tournament_k']} "
                      f"(time={result['runtime']:.1f}s, decisions={result['decisions']})")
            
            # Best configuration
            best = successful[0]
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   Population Size: {best['pop_size']}")
            print(f"   Crossover Rate: {best['crossover_rate']:.2f}")
            print(f"   Mutation Rate: {best['mutation_rate']:.3f}")
            print(f"   Tournament K: {best['tournament_k']}")
            print(f"   Runtime: {best['runtime']:.1f} seconds")
            print(f"   Decisions: {best['decisions']}")
        
        if failed:
            print(f"\nFailed configurations (showing best 3 by conflicts):")
            failed.sort(key=lambda x: x['final_conflicts'])
            for i, result in enumerate(failed[:3], 1):
                print(f"  {i}. pop={result['pop_size']}, cross={result['crossover_rate']:.2f}, "
                      f"mut={result['mutation_rate']:.3f}, k={result['tournament_k']} "
                      f"(conflicts={result['final_conflicts']}, time={result['runtime']:.1f}s)")
    
    def save_results(self, filename: str = "quick_ga_tuning_results.json"):
        """Save results to JSON file"""
        if not self.results:
            print("No results to save.")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filename}")


def main():
    """Main function"""
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Choose puzzle
    puzzle_path = "puzzles/Easy-P1.txt"  # Start with easy puzzle
    
    print("Quick GA Hyperparameter Tuning")
    print("="*40)
    print(f"Puzzle: {puzzle_path}")
    
    # Initialize tuner
    tuner = QuickGATuner(puzzle_path)
    
    # Choose method
    print("\nChoose tuning method:")
    print("1. Quick Grid Search (8 predefined configurations)")
    print("2. Random Search (20 random configurations)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        timeout = int(input("Timeout per configuration in seconds (default 60): ") or "60")
        results = tuner.quick_grid_search(timeout)
    elif choice == "2":
        n_trials = int(input("Number of random trials (default 20): ") or "20")
        timeout = int(input("Timeout per configuration in seconds (default 60): ") or "60")
        results = tuner.random_search(n_trials, timeout)
    else:
        print("Invalid choice. Running quick grid search...")
        results = tuner.quick_grid_search(60)
    
    # Analyze results
    tuner.analyze_results()
    
    # Save results
    tuner.save_results()
    
    print("\nTuning complete! Check the results above for the best configuration.")


if __name__ == "__main__":
    main()
