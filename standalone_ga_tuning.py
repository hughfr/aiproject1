#!/usr/bin/env python3
"""
Standalone GA Hyperparameter Tuning Script

A self-contained version that includes all necessary GA functions
without requiring imports from the Jupyter notebook.
"""

import numpy as np
import time
import random
import json
import os
import re
from typing import Dict, List, Tuple

class StandaloneGATuner:
    """Standalone GA hyperparameter tuner with all necessary functions"""
    
    def __init__(self, puzzle_path: str):
        self.puzzle_path = puzzle_path
        self.results = []
        self.decisions = 0
        
    def inc_decisions(self, n=1):
        """Increment decision counter"""
        self.decisions += n
    
    def reset_decisions(self):
        """Reset decision counter"""
        self.decisions = 0
    
    def read_puzzle_txt(self, path):
        """Read 9x9 Sudoku puzzle from text file with BOM support"""
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) != 9:
                raise ValueError(f"Expected 9 lines, got {len(lines)}")
            
            grid = []
            for i, line in enumerate(lines):
                if ',' in line:
                    values = [v.strip() for v in line.split(',')]
                    if len(values) != 9:
                        raise ValueError(f"Line {i+1} has {len(values)} values, expected 9")
                else:
                    if not re.match(r'^[0-9?]{9}$', line):
                        raise ValueError(f"Line {i+1} contains invalid characters: {line}")
                    values = list(line)
                
                row = []
                for value in values:
                    if value == '?':
                        row.append(0)
                    else:
                        row.append(int(value))
                grid.append(row)
            
            return np.array(grid, dtype=int)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {path}")
        except Exception as e:
            raise ValueError(f"Error reading puzzle file: {e}")
    
    def sudoku_from_grid(self, grid):
        """Create Sudoku object from 9x9 grid"""
        if not isinstance(grid, np.ndarray) or grid.shape != (9, 9):
            raise ValueError("Grid must be 9x9 numpy array")
        
        if not np.all((grid >= 0) & (grid <= 9)):
            raise ValueError("Grid values must be in range 0-9")
        
        givens_mask = grid > 0
        
        return {
            'grid': grid.copy(),
            'givens_mask': givens_mask
        }
    
    def is_solved(self, sud):
        """Check if Sudoku puzzle is completely solved"""
        grid = sud['grid']
        
        if np.any(grid == 0):
            return False
        
        # Check rows
        for row in grid:
            if len(set(row)) != 9 or not all(1 <= x <= 9 for x in row):
                return False
        
        # Check columns
        for col in grid.T:
            if len(set(col)) != 9 or not all(1 <= x <= 9 for x in col):
                return False
        
        # Check 3x3 boxes
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = grid[i:i+3, j:j+3].flatten()
                if len(set(box)) != 9 or not all(1 <= x <= 9 for x in box):
                    return False
        
        return True
    
    def total_conflicts(self, sud):
        """Count total number of constraint violations"""
        grid = sud['grid']
        conflicts = 0
        
        # Check rows
        for r in range(9):
            row_values = [grid[r, c] for c in range(9) if grid[r, c] != 0]
            conflicts += len(row_values) - len(set(row_values))
        
        # Check columns
        for c in range(9):
            col_values = [grid[r, c] for r in range(9) if grid[r, c] != 0]
            conflicts += len(col_values) - len(set(col_values))
        
        # Check 3x3 boxes
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box_values = []
                for r in range(i, i+3):
                    for c in range(j, j+3):
                        if grid[r, c] != 0:
                            box_values.append(grid[r, c])
                conflicts += len(box_values) - len(set(box_values))
        
        return conflicts
    
    def random_complete_assignment(self, sud):
        """Create random complete assignment respecting givens"""
        C = sud['grid'].copy()
        givens_mask = sud['givens_mask']
        
        for br in range(3):
            for bc in range(3):
                # Get given values in this box
                given_values = set()
                cells = []
                for r in range(br*3, (br+1)*3):
                    for c in range(bc*3, (bc+1)*3):
                        if givens_mask[r, c]:
                            given_values.add(C[r, c])
                        else:
                            cells.append((r, c))
                
                # Missing values for this box
                missing = set(range(1, 10)) - given_values
                
                # Place missing values randomly in non-given cells
                missing_list = list(missing)
                random.shuffle(missing_list)
                
                for i, (r, c) in enumerate(cells):
                    if i < len(missing_list):
                        C[r, c] = missing_list[i]
        
        return {'grid': C, 'givens_mask': givens_mask}
    
    def ga_fitness(self, individual):
        """Fitness function: higher is better"""
        self.inc_decisions(1)  # Count each fitness evaluation
        return -self.total_conflicts(individual)
    
    def ga_tournament_select(self, population, fitnesses, k):
        """Tournament selection: sample k individuals, return best"""
        tournament_indices = random.sample(range(len(population)), k)
        best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
        return {'grid': population[best_idx]['grid'].copy(), 'givens_mask': population[best_idx]['givens_mask']}
    
    def ga_crossover(self, p1, p2):
        """Block-based crossover that preserves box constraints"""
        child1 = {'grid': p1['grid'].copy(), 'givens_mask': p1['givens_mask']}
        child2 = {'grid': p2['grid'].copy(), 'givens_mask': p2['givens_mask']}
        
        # Randomly choose which 3x3 blocks to copy from each parent
        for br in range(3):
            for bc in range(3):
                if random.random() < 0.5:
                    # Copy block from p1 to child1, p2 to child2
                    for r in range(br*3, (br+1)*3):
                        for c in range(bc*3, (bc+1)*3):
                            child1['grid'][r, c] = p1['grid'][r, c]
                            child2['grid'][r, c] = p2['grid'][r, c]
                else:
                    # Copy block from p2 to child1, p1 to child2
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
    
    def ga_mutate(self, individual, rate, sud):
        """Mutate using multiple strategies"""
        mutated = {'grid': individual['grid'].copy(), 'givens_mask': individual['givens_mask']}
        
        # Apply mutation with probability rate
        if random.random() < rate:
            # Strategy 1: Swap within same box (70% of the time)
            if random.random() < 0.7:
                # Pick random box
                br, bc = random.randint(0, 2), random.randint(0, 2)
                
                # Get non-given cells in this box
                cells = []
                for r in range(br*3, (br+1)*3):
                    for c in range(bc*3, (bc+1)*3):
                        if not sud['givens_mask'][r, c]:
                            cells.append((r, c))
                
                if len(cells) >= 2:
                    # Pick two random cells and swap their values
                    cell1, cell2 = random.sample(cells, 2)
                    r1, c1 = cell1
                    r2, c2 = cell2
                    
                    mutated['grid'][r1, c1], mutated['grid'][r2, c2] = mutated['grid'][r2, c2], mutated['grid'][r1, c1]
                    self.inc_decisions(1)
            
            # Strategy 2: Swap across boxes (30% of the time)
            else:
                # Find two non-given cells in different boxes and swap them
                non_given_cells = []
                for r in range(9):
                    for c in range(9):
                        if not sud['givens_mask'][r, c]:
                            non_given_cells.append((r, c))
                
                if len(non_given_cells) >= 2:
                    cell1, cell2 = random.sample(non_given_cells, 2)
                    r1, c1 = cell1
                    r2, c2 = cell2
                    
                    # Check if they're in different boxes
                    box1 = (r1 // 3, c1 // 3)
                    box2 = (r2 // 3, c2 // 3)
                    
                    if box1 != box2:
                        mutated['grid'][r1, c1], mutated['grid'][r2, c2] = mutated['grid'][r2, c2], mutated['grid'][r1, c1]
                        self.inc_decisions(1)
        
        return mutated
    
    def solve_ga_quick(self, sud, pop_size, crossover_rate, mutation_rate, 
                      tournament_k, max_generations, timeout):
        """Quick GA implementation with early stopping"""
        start_time = time.time()
        
        # Initialize population
        population = [self.random_complete_assignment(sud) for _ in range(pop_size)]
        
        for gen in range(max_generations):
            # Check timeout
            if time.time() - start_time > timeout:
                break
                
            # Check if any individual is solved
            for individual in population:
                if self.total_conflicts(individual) == 0:
                    return individual
            
            # Calculate fitnesses
            fitnesses = [self.ga_fitness(x) for x in population]
            
            # Create offspring
            offspring = []
            
            while len(offspring) < pop_size:
                # Selection
                p1 = self.ga_tournament_select(population, fitnesses, tournament_k)
                p2 = self.ga_tournament_select(population, fitnesses, tournament_k)
                
                # Crossover
                if random.random() < crossover_rate:
                    c1, c2 = self.ga_crossover(p1, p2)
                else:
                    c1 = {'grid': p1['grid'].copy(), 'givens_mask': p1['givens_mask']}
                    c2 = {'grid': p2['grid'].copy(), 'givens_mask': p2['givens_mask']}
                
                # Mutation
                c1 = self.ga_mutate(c1, mutation_rate, sud)
                c2 = self.ga_mutate(c2, mutation_rate, sud)
                
                offspring.append(c1)
                if len(offspring) < pop_size:
                    offspring.append(c2)
            
            # Replace population (no elitism)
            population = offspring
        
        # Return best found
        fitnesses = [self.ga_fitness(x) for x in population]
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        return population[best_idx]
    
    def evaluate_config(self, pop_size: int, crossover_rate: float, 
                       mutation_rate: float, tournament_k: int, 
                       max_generations: int = 500, timeout: int = 120) -> Dict:
        """Evaluate a single parameter configuration"""
        start_time = time.time()
        
        try:
            # Read puzzle
            grid = self.read_puzzle_txt(self.puzzle_path)
            sud = self.sudoku_from_grid(grid)
            
            # Reset decision counter
            self.reset_decisions()
            
            # Run GA with early stopping
            solution = self.solve_ga_quick(sud, pop_size, crossover_rate, 
                                        mutation_rate, tournament_k, 
                                        max_generations, timeout)
            
            runtime = time.time() - start_time
            
            # Analyze results
            if solution is not None:
                success = self.is_solved(solution)
                final_conflicts = self.total_conflicts(solution)
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
                'decisions': self.decisions,
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
        print("GA HYPERPARAMETER TUNING RESULTS")
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
    
    def save_results(self, filename: str = "standalone_ga_tuning_results.json"):
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
    
    print("Standalone GA Hyperparameter Tuning")
    print("="*40)
    print(f"Puzzle: {puzzle_path}")
    
    # Check if puzzle file exists
    if not os.path.exists(puzzle_path):
        print(f"Error: Puzzle file '{puzzle_path}' not found!")
        print("Available puzzle files:")
        if os.path.exists("puzzles"):
            for f in os.listdir("puzzles"):
                if f.endswith(".txt"):
                    print(f"  - puzzles/{f}")
        return
    
    # Initialize tuner
    tuner = StandaloneGATuner(puzzle_path)
    
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
