#!/usr/bin/env python3
"""
Genetic Algorithm Hyperparameter Tuning Script for Sudoku Solver

This script efficiently tunes GA hyperparameters using:
- Parallel processing for multiple parameter combinations
- Early stopping to avoid long runs
- Statistical analysis of results
- Grid search and random search options
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
import os
from itertools import product
import random
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import your GA implementation
from sudoku_solver2 import (
    read_puzzle_txt, sudoku_from_grid, solve_ga, 
    total_conflicts, is_solved, reset_decisions, decisions
)

@dataclass
class GAParams:
    """Container for GA hyperparameters"""
    pop_size: int
    crossover_rate: float
    mutation_rate: float
    tournament_k: int
    max_generations: int = 1000  # Reduced for tuning

@dataclass
class TuningResult:
    """Container for tuning results"""
    params: GAParams
    success: bool
    generations_to_solve: int
    final_conflicts: int
    decisions: int
    runtime: float
    puzzle_path: str

class GAHyperparameterTuner:
    """Main class for GA hyperparameter tuning"""
    
    def __init__(self, puzzle_paths: List[str], max_workers: int = None):
        self.puzzle_paths = puzzle_paths
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.results = []
        
    def evaluate_single_config(self, params: GAParams, puzzle_path: str, 
                             timeout: int = 300) -> TuningResult:
        """Evaluate a single parameter configuration on one puzzle"""
        start_time = time.time()
        
        try:
            # Read puzzle
            grid = read_puzzle_txt(puzzle_path)
            sud = sudoku_from_grid(grid)
            
            # Reset decision counter
            reset_decisions()
            
            # Run GA with timeout
            solution = self._run_ga_with_timeout(
                sud, params, timeout
            )
            
            runtime = time.time() - start_time
            
            # Analyze results
            if solution is not None:
                success = is_solved(solution)
                final_conflicts = total_conflicts(solution)
                generations_to_solve = params.max_generations  # Will be updated if solved early
            else:
                success = False
                final_conflicts = float('inf')
                generations_to_solve = params.max_generations
            
            return TuningResult(
                params=params,
                success=success,
                generations_to_solve=generations_to_solve,
                final_conflicts=final_conflicts,
                decisions=decisions,
                runtime=runtime,
                puzzle_path=puzzle_path
            )
            
        except Exception as e:
            print(f"Error evaluating {params} on {puzzle_path}: {e}")
            return TuningResult(
                params=params,
                success=False,
                generations_to_solve=params.max_generations,
                final_conflicts=float('inf'),
                decisions=0,
                runtime=time.time() - start_time,
                puzzle_path=puzzle_path
            )
    
    def _run_ga_with_timeout(self, sud, params: GAParams, timeout: int):
        """Run GA with timeout and early stopping"""
        # Modified GA that stops early when solution is found
        return self._solve_ga_early_stop(sud, params, timeout)
    
    def _solve_ga_early_stop(self, sud, params: GAParams, timeout: int):
        """Modified GA with early stopping and timeout"""
        import time
        start_time = time.time()
        
        # Initialize population
        population = [self._random_complete_assignment(sud) for _ in range(params.pop_size)]
        
        for gen in range(params.max_generations):
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
            
            while len(offspring) < params.pop_size:
                # Selection
                p1 = self._ga_tournament_select(population, fitnesses, params.tournament_k)
                p2 = self._ga_tournament_select(population, fitnesses, params.tournament_k)
                
                # Crossover
                if random.random() < params.crossover_rate:
                    c1, c2 = self._ga_crossover(p1, p2)
                else:
                    c1 = {'grid': p1['grid'].copy(), 'givens_mask': p1['givens_mask']}
                    c2 = {'grid': p2['grid'].copy(), 'givens_mask': p2['givens_mask']}
                
                # Mutation
                c1 = self._ga_mutate(c1, params.mutation_rate, sud)
                c2 = self._ga_mutate(c2, params.mutation_rate, sud)
                
                offspring.append(c1)
                if len(offspring) < params.pop_size:
                    offspring.append(c2)
            
            # Replace population (no elitism)
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
        """Mutate using multiple strategies"""
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
    
    def grid_search(self, param_grid: Dict[str, List], timeout: int = 300) -> List[TuningResult]:
        """Perform grid search over parameter combinations"""
        print(f"Starting grid search with {self.max_workers} workers...")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"Total combinations to test: {len(combinations)}")
        print(f"Puzzles to test: {len(self.puzzle_paths)}")
        print(f"Total evaluations: {len(combinations) * len(self.puzzle_paths)}")
        
        # Create tasks
        tasks = []
        for params_tuple in combinations:
            params_dict = dict(zip(param_names, params_tuple))
            params = GAParams(**params_dict)
            
            for puzzle_path in self.puzzle_paths:
                tasks.append((params, puzzle_path, timeout))
        
        # Run evaluations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.evaluate_single_config, params, puzzle_path, timeout): (params, puzzle_path)
                for params, puzzle_path, timeout in tasks
            }
            
            # Collect results
            completed = 0
            total = len(tasks)
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"Completed {completed}/{total} evaluations ({completed/total*100:.1f}%)")
                        
                except Exception as e:
                    params, puzzle_path = future_to_task[future]
                    print(f"Task failed for {params} on {puzzle_path}: {e}")
        
        self.results = results
        return results
    
    def random_search(self, n_trials: int, param_ranges: Dict[str, Tuple], 
                     timeout: int = 300) -> List[TuningResult]:
        """Perform random search over parameter space"""
        print(f"Starting random search with {n_trials} trials...")
        
        # Generate random parameter combinations
        tasks = []
        for _ in range(n_trials):
            params_dict = {}
            for param_name, (min_val, max_val, param_type) in param_ranges.items():
                if param_type == int:
                    params_dict[param_name] = random.randint(min_val, max_val)
                elif param_type == float:
                    params_dict[param_name] = random.uniform(min_val, max_val)
            
            params = GAParams(**params_dict)
            
            for puzzle_path in self.puzzle_paths:
                tasks.append((params, puzzle_path, timeout))
        
        print(f"Total evaluations: {len(tasks)}")
        
        # Run evaluations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.evaluate_single_config, params, puzzle_path, timeout): (params, puzzle_path)
                for params, puzzle_path, timeout in tasks
            }
            
            completed = 0
            total = len(tasks)
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"Completed {completed}/{total} evaluations ({completed/total*100:.1f}%)")
                        
                except Exception as e:
                    params, puzzle_path = future_to_task[future]
                    print(f"Task failed for {params} on {puzzle_path}: {e}")
        
        self.results = results
        return results
    
    def analyze_results(self) -> pd.DataFrame:
        """Analyze and summarize results"""
        if not self.results:
            print("No results to analyze. Run grid_search or random_search first.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for result in self.results:
            data.append({
                'pop_size': result.params.pop_size,
                'crossover_rate': result.params.crossover_rate,
                'mutation_rate': result.params.mutation_rate,
                'tournament_k': result.params.tournament_k,
                'success': result.success,
                'generations_to_solve': result.generations_to_solve,
                'final_conflicts': result.final_conflicts,
                'decisions': result.decisions,
                'runtime': result.runtime,
                'puzzle': os.path.basename(result.puzzle_path)
            })
        
        df = pd.DataFrame(data)
        
        # Summary statistics
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING RESULTS")
        print("="*60)
        
        print(f"\nTotal evaluations: {len(df)}")
        print(f"Successful solves: {df['success'].sum()}")
        print(f"Success rate: {df['success'].mean()*100:.1f}%")
        
        if df['success'].any():
            successful = df[df['success']]
            print(f"\nSuccessful configurations:")
            print(f"  Average generations to solve: {successful['generations_to_solve'].mean():.1f}")
            print(f"  Average decisions: {successful['decisions'].mean():.0f}")
            print(f"  Average runtime: {successful['runtime'].mean():.1f}s")
            
            # Best configurations
            print(f"\nTop 5 configurations by generations to solve:")
            top_configs = successful.nsmallest(5, 'generations_to_solve')
            for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                print(f"  {i}. pop={row['pop_size']}, cross={row['crossover_rate']:.2f}, "
                      f"mut={row['mutation_rate']:.3f}, k={row['tournament_k']} "
                      f"(gens={row['generations_to_solve']}, time={row['runtime']:.1f}s)")
        
        return df
    
    def save_results(self, filename: str = "ga_tuning_results.json"):
        """Save results to JSON file"""
        if not self.results:
            print("No results to save.")
            return
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'params': {
                    'pop_size': result.params.pop_size,
                    'crossover_rate': result.params.crossover_rate,
                    'mutation_rate': result.params.mutation_rate,
                    'tournament_k': result.params.tournament_k,
                    'max_generations': result.params.max_generations
                },
                'success': result.success,
                'generations_to_solve': result.generations_to_solve,
                'final_conflicts': result.final_conflicts,
                'decisions': result.decisions,
                'runtime': result.runtime,
                'puzzle_path': result.puzzle_path
            })
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def plot_results(self, df: pd.DataFrame):
        """Create visualization plots"""
        if df.empty:
            print("No data to plot.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('GA Hyperparameter Tuning Results', fontsize=16)
        
        # 1. Success rate by parameter
        params = ['pop_size', 'crossover_rate', 'mutation_rate', 'tournament_k']
        for i, param in enumerate(params):
            if i < 4:
                ax = axes[i//2, i%2] if i < 4 else axes[1, 2]
                
                # Group by parameter value and calculate success rate
                success_by_param = df.groupby(param)['success'].agg(['mean', 'count']).reset_index()
                success_by_param = success_by_param[success_by_param['count'] >= 3]  # Filter low counts
                
                if not success_by_param.empty:
                    ax.bar(success_by_param[param], success_by_param['mean'])
                    ax.set_title(f'Success Rate by {param}')
                    ax.set_xlabel(param)
                    ax.set_ylabel('Success Rate')
                    ax.set_ylim(0, 1)
        
        # 2. Runtime vs Success
        ax = axes[1, 2]
        successful = df[df['success']]
        failed = df[~df['success']]
        
        if not successful.empty:
            ax.scatter(successful['runtime'], successful['generations_to_solve'], 
                      alpha=0.6, label='Successful', color='green')
        if not failed.empty:
            ax.scatter(failed['runtime'], [1000]*len(failed), 
                      alpha=0.6, label='Failed', color='red')
        
        ax.set_xlabel('Runtime (s)')
        ax.set_ylabel('Generations to Solve')
        ax.set_title('Runtime vs Performance')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('ga_tuning_plots.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run hyperparameter tuning"""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define puzzle paths to test on
    puzzle_paths = [
        "puzzles/Easy-P1.txt",
        "puzzles/Easy-P2.txt",
        # Add more puzzles as needed
    ]
    
    # Initialize tuner
    tuner = GAHyperparameterTuner(puzzle_paths, max_workers=4)
    
    # Option 1: Grid Search (comprehensive but slower)
    print("Option 1: Grid Search")
    param_grid = {
        'pop_size': [50, 100, 200, 300],
        'crossover_rate': [0.7, 0.8, 0.9, 0.95],
        'mutation_rate': [0.01, 0.03, 0.05, 0.1],
        'tournament_k': [3, 5, 7, 10]
    }
    
    # Option 2: Random Search (faster, good for exploration)
    print("Option 2: Random Search")
    param_ranges = {
        'pop_size': (50, 300, int),
        'crossover_rate': (0.6, 0.95, float),
        'mutation_rate': (0.01, 0.1, float),
        'tournament_k': (3, 10, int)
    }
    
    # Choose which method to run
    method = input("Choose method (1: Grid Search, 2: Random Search): ").strip()
    
    if method == "1":
        print("Running Grid Search...")
        results = tuner.grid_search(param_grid, timeout=180)  # 3 minute timeout
    elif method == "2":
        n_trials = int(input("Number of random trials (default 50): ") or "50")
        print(f"Running Random Search with {n_trials} trials...")
        results = tuner.random_search(n_trials, param_ranges, timeout=180)
    else:
        print("Invalid choice. Running Random Search with 20 trials...")
        results = tuner.random_search(20, param_ranges, timeout=180)
    
    # Analyze results
    df = tuner.analyze_results()
    
    # Save results
    tuner.save_results()
    
    # Create plots
    if not df.empty:
        tuner.plot_results(df)
    
    print("\nTuning complete!")


if __name__ == "__main__":
    main()
