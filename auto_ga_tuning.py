#!/usr/bin/env python3
"""
Automated GA Hyperparameter Tuning Script

Runs continuously over a long period, automatically testing different parameter
combinations and saving the best results. Designed to run unattended.
"""

import numpy as np
import time
import random
import json
import os
import re
from typing import Dict, List, Tuple
from datetime import datetime
import signal
import sys

class AutoGATuner:
    """Automated GA hyperparameter tuner that runs continuously"""
    
    def __init__(self, puzzle_path: str, results_file: str = "auto_ga_results.json", difficulty: str = "easy"):
        self.puzzle_path = puzzle_path
        self.results_file = results_file
        self.difficulty = difficulty
        self.results = []
        self.best_result = None
        self.decisions = 0
        self.start_time = time.time()
        self.total_tests = 0
        self.successful_tests = 0
        
        # Load existing results if available
        self.load_existing_results()
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n\nReceived signal {signum}. Saving results and shutting down...")
        self.save_results()
        print(f"Results saved. Total tests: {self.total_tests}, Successful: {self.successful_tests}")
        sys.exit(0)
    
    def load_existing_results(self):
        """Load existing results from file"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    self.results = data.get('results', [])
                    self.best_result = data.get('best_result')
                    self.total_tests = data.get('total_tests', 0)
                    self.successful_tests = data.get('successful_tests', 0)
                print(f"Loaded {len(self.results)} existing results")
                if self.best_result:
                    print(f"Current best: {self.best_result['runtime']:.1f}s, {self.best_result['decisions']} decisions")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                self.results = []
    
    def save_results(self):
        """Save results to file"""
        data = {
            'results': self.results,
            'best_result': self.best_result,
            'total_tests': self.total_tests,
            'successful_tests': self.successful_tests,
            'last_updated': datetime.now().isoformat(),
            'puzzle_path': self.puzzle_path
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def inc_decisions(self, n=1):
        """Increment decision counter"""
        self.decisions += n
    
    def reset_decisions(self):
        """Reset decision counter"""
        self.decisions = 0
    
    def read_puzzle_txt(self, path):
        """Read 9x9 Sudoku puzzle from text file"""
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
    
    def ga_fitness(self, individual):
        """Fitness function: higher is better"""
        self.inc_decisions(1)
        return -self.total_conflicts(individual)
    
    def ga_tournament_select(self, population, fitnesses, k):
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), k)
        best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
        return {'grid': population[best_idx]['grid'].copy(), 'givens_mask': population[best_idx]['givens_mask']}
    
    def ga_crossover(self, p1, p2):
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
    
    def ga_mutate(self, individual, rate, sud):
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
                    self.inc_decisions(1)
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
                        self.inc_decisions(1)
        
        return mutated
    
    def solve_ga_auto(self, sud, pop_size, crossover_rate, mutation_rate, 
                     tournament_k, max_generations, timeout):
        """GA implementation with early stopping for auto-tuning"""
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
            
            # Replace population
            population = offspring
        
        # Return best found
        fitnesses = [self.ga_fitness(x) for x in population]
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        return population[best_idx]
    
    def generate_random_params(self, puzzle_difficulty="easy"):
        """Generate random parameter combination based on puzzle difficulty"""
        # Population size: favor larger populations for harder puzzles
        if puzzle_difficulty == "easy":
            pop_size = random.choices([100, 150, 200, 300, 400], 
                                    weights=[2, 3, 4, 3, 1])[0]
        elif puzzle_difficulty == "medium":
            pop_size = random.choices([200, 300, 400, 500, 600], 
                                    weights=[2, 3, 3, 2, 1])[0]
        else:  # hard/evil
            pop_size = random.choices([300, 400, 500, 600, 800], 
                                    weights=[1, 2, 3, 3, 2])[0]
        
        # Crossover rate: favor higher rates
        crossover_rate = random.uniform(0.7, 0.95)
        
        # Mutation rate: explore different ranges, favor higher for harder puzzles
        if puzzle_difficulty == "easy":
            mutation_rate = random.choices([
                random.uniform(0.01, 0.03),  # Low mutation
                random.uniform(0.03, 0.07),  # Medium mutation
                random.uniform(0.07, 0.12)   # High mutation
            ], weights=[2, 3, 1])[0]
        else:  # medium/hard/evil
            mutation_rate = random.choices([
                random.uniform(0.03, 0.06),  # Medium mutation
                random.uniform(0.06, 0.10),  # High mutation
                random.uniform(0.10, 0.15)   # Very high mutation
            ], weights=[2, 3, 2])[0]
        
        # Tournament size: favor larger for harder puzzles
        if puzzle_difficulty == "easy":
            tournament_k = random.choices([3, 5, 7, 10], 
                                        weights=[1, 3, 3, 2])[0]
        else:  # medium/hard/evil
            tournament_k = random.choices([5, 7, 10, 15, 20], 
                                        weights=[2, 3, 3, 2, 1])[0]
        
        # Max generations: much higher for harder puzzles
        if puzzle_difficulty == "easy":
            max_generations = random.randint(500, 1500)
        elif puzzle_difficulty == "medium":
            max_generations = random.randint(1000, 3000)
        else:  # hard/evil
            max_generations = random.randint(2000, 5000)
        
        # Timeout: longer for harder puzzles and larger populations
        base_timeout = 120 if puzzle_difficulty == "easy" else 300 if puzzle_difficulty == "medium" else 600
        timeout = min(base_timeout, base_timeout + pop_size // 20)
        
        return pop_size, crossover_rate, mutation_rate, tournament_k, max_generations, timeout
    
    def evaluate_config_auto(self, pop_size, crossover_rate, mutation_rate, 
                           tournament_k, max_generations, timeout):
        """Evaluate a single parameter configuration"""
        start_time = time.time()
        
        try:
            # Read puzzle
            grid = self.read_puzzle_txt(self.puzzle_path)
            sud = self.sudoku_from_grid(grid)
            
            # Reset decision counter
            self.reset_decisions()
            
            # Run GA
            solution = self.solve_ga_auto(sud, pop_size, crossover_rate, 
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
            
            result = {
                'pop_size': pop_size,
                'crossover_rate': crossover_rate,
                'mutation_rate': mutation_rate,
                'tournament_k': tournament_k,
                'max_generations': max_generations,
                'success': success,
                'final_conflicts': final_conflicts,
                'decisions': self.decisions,
                'runtime': runtime,
                'timestamp': datetime.now().isoformat(),
                'test_number': self.total_tests + 1
            }
            
            return result
            
        except Exception as e:
            return {
                'pop_size': pop_size,
                'crossover_rate': crossover_rate,
                'mutation_rate': mutation_rate,
                'tournament_k': tournament_k,
                'max_generations': max_generations,
                'success': False,
                'final_conflicts': float('inf'),
                'decisions': 0,
                'runtime': time.time() - start_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'test_number': self.total_tests + 1
            }
    
    def update_best_result(self, result):
        """Update best result if this one is better"""
        if not result['success']:
            return False
        
        if self.best_result is None:
            self.best_result = result
            return True
        
        # Prefer faster solutions, then fewer decisions
        if (result['runtime'] < self.best_result['runtime'] or
            (result['runtime'] == self.best_result['runtime'] and 
             result['decisions'] < self.best_result['decisions'])):
            self.best_result = result
            return True
        
        return False
    
    def print_status(self, result):
        """Print current status"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Test #{result['test_number']} "
              f"(Running {hours}h {minutes}m)")
        print(f"Params: pop={result['pop_size']}, cross={result['crossover_rate']:.2f}, "
              f"mut={result['mutation_rate']:.3f}, k={result['tournament_k']}")
        
        if result['success']:
            print(f"🎉 SUCCESS! Runtime: {result['runtime']:.1f}s, Decisions: {result['decisions']}")
            if self.update_best_result(result):
                print(f"🏆 NEW BEST! Previous best: {self.best_result.get('runtime', 'N/A'):.1f}s")
        else:
            print(f"❌ Failed. Conflicts: {result['final_conflicts']}, Runtime: {result['runtime']:.1f}s")
        
        print(f"Total tests: {self.total_tests}, Successful: {self.successful_tests} "
              f"({self.successful_tests/max(1,self.total_tests)*100:.1f}%)")
        
        if self.best_result:
            print(f"Best so far: {self.best_result['runtime']:.1f}s, "
                  f"{self.best_result['decisions']} decisions")
    
    def run_continuous(self, max_duration_hours=None, save_interval=10):
        """Run continuous tuning"""
        print("🤖 Starting Automated GA Hyperparameter Tuning")
        print("="*60)
        print(f"Puzzle: {self.puzzle_path}")
        print(f"Results file: {self.results_file}")
        if max_duration_hours:
            print(f"Max duration: {max_duration_hours} hours")
        print("Press Ctrl+C to stop gracefully")
        print("="*60)
        
        if not os.path.exists(self.puzzle_path):
            print(f"Error: Puzzle file '{self.puzzle_path}' not found!")
            return
        
        start_time = time.time()
        last_save = time.time()
        
        try:
            while True:
                # Check if we've exceeded max duration
                if max_duration_hours:
                    elapsed_hours = (time.time() - start_time) / 3600
                    if elapsed_hours >= max_duration_hours:
                        print(f"\n⏰ Reached maximum duration of {max_duration_hours} hours")
                        break
                
                # Generate random parameters based on difficulty
                pop_size, crossover_rate, mutation_rate, tournament_k, max_generations, timeout = self.generate_random_params(self.difficulty)
                
                # Evaluate configuration
                result = self.evaluate_config_auto(pop_size, crossover_rate, mutation_rate, 
                                                tournament_k, max_generations, timeout)
                
                # Update counters
                self.total_tests += 1
                if result['success']:
                    self.successful_tests += 1
                
                # Store result
                self.results.append(result)
                
                # Print status
                self.print_status(result)
                
                # Save results periodically
                if time.time() - last_save > save_interval * 60:  # Save every N minutes
                    self.save_results()
                    last_save = time.time()
                    print(f"💾 Results saved to {self.results_file}")
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Stopped by user")
        
        # Final save
        self.save_results()
        print(f"\n📊 Final Results:")
        print(f"Total tests: {self.total_tests}")
        print(f"Successful: {self.successful_tests}")
        print(f"Success rate: {self.successful_tests/max(1,self.total_tests)*100:.1f}%")
        
        if self.best_result:
            print(f"\n🏆 Best Configuration:")
            print(f"   Population: {self.best_result['pop_size']}")
            print(f"   Crossover: {self.best_result['crossover_rate']:.2f}")
            print(f"   Mutation: {self.best_result['mutation_rate']:.3f}")
            print(f"   Tournament K: {self.best_result['tournament_k']}")
            print(f"   Runtime: {self.best_result['runtime']:.1f}s")
            print(f"   Decisions: {self.best_result['decisions']}")
        
        print(f"\nResults saved to: {self.results_file}")


def main():
    """Main function"""
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    print("Automated GA Hyperparameter Tuning")
    print("="*40)
    
    # Show available puzzles
    if not os.path.exists("puzzles"):
        print("Error: puzzles directory not found!")
        return
    
    available_puzzles = [f for f in os.listdir("puzzles") if f.endswith(".txt")]
    if not available_puzzles:
        print("Error: No puzzle files found in puzzles directory!")
        return
    
    print("Available puzzles:")
    for i, puzzle in enumerate(available_puzzles, 1):
        print(f"  {i}. {puzzle}")
    
    # Get puzzle selection
    print("\nChoose puzzle difficulty:")
    print("1. Easy puzzles (P1-P4)")
    print("2. Medium puzzles (P1-P4)")
    print("3. Hard puzzles (P1-P4)")
    print("4. Evil puzzles (P1-P4)")
    print("5. Custom puzzle")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        puzzle_files = [f for f in available_puzzles if f.startswith("Easy-")]
        difficulty = "easy"
    elif choice == "2":
        puzzle_files = [f for f in available_puzzles if f.startswith("Med-")]
        difficulty = "medium"
    elif choice == "3":
        puzzle_files = [f for f in available_puzzles if f.startswith("Hard-")]
        difficulty = "hard"
    elif choice == "4":
        puzzle_files = [f for f in available_puzzles if f.startswith("Evil-")]
        difficulty = "evil"
    elif choice == "5":
        puzzle_name = input("Enter puzzle filename (e.g., Easy-P1.txt): ").strip()
        if puzzle_name in available_puzzles:
            puzzle_files = [puzzle_name]
            difficulty = "custom"
        else:
            print(f"Error: {puzzle_name} not found!")
            return
    else:
        print("Invalid choice. Using Easy puzzles.")
        puzzle_files = [f for f in available_puzzles if f.startswith("Easy-")]
        difficulty = "easy"
    
    if not puzzle_files:
        print(f"Error: No {difficulty} puzzles found!")
        return
    
    # Select specific puzzle
    if len(puzzle_files) > 1:
        print(f"\nAvailable {difficulty} puzzles:")
        for i, puzzle in enumerate(puzzle_files, 1):
            print(f"  {i}. {puzzle}")
        
        puzzle_choice = input(f"Choose puzzle (1-{len(puzzle_files)}): ").strip()
        try:
            puzzle_idx = int(puzzle_choice) - 1
            if 0 <= puzzle_idx < len(puzzle_files):
                puzzle_path = f"puzzles/{puzzle_files[puzzle_idx]}"
            else:
                print("Invalid choice. Using first puzzle.")
                puzzle_path = f"puzzles/{puzzle_files[0]}"
        except ValueError:
            print("Invalid choice. Using first puzzle.")
            puzzle_path = f"puzzles/{puzzle_files[0]}"
    else:
        puzzle_path = f"puzzles/{puzzle_files[0]}"
    
    # Generate results filename based on puzzle
    puzzle_name = puzzle_files[0].replace(".txt", "")
    results_file = f"auto_ga_results_{puzzle_name}.json"
    
    print(f"\nConfiguration:")
    print(f"Puzzle: {puzzle_path}")
    print(f"Difficulty: {difficulty}")
    print(f"Results file: {results_file}")
    
    # Get user preferences
    max_hours = input("Maximum duration in hours (press Enter for unlimited): ").strip()
    max_hours = int(max_hours) if max_hours else None
    
    save_interval = input("Save interval in minutes (default 10): ").strip()
    save_interval = int(save_interval) if save_interval else 10
    
    # Initialize and run
    tuner = AutoGATuner(puzzle_path, results_file, difficulty)
    tuner.run_continuous(max_duration_hours=max_hours, save_interval=save_interval)


if __name__ == "__main__":
    main()
