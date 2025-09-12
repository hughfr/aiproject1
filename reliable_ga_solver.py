#!/usr/bin/env python3
"""
Reliable GA Solver - Multiple runs with different seeds to ensure success
"""

import numpy as np
import time
import random
import re
from typing import List, Tuple, Optional

def read_puzzle_txt(path):
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

def sudoku_from_grid(grid):
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

def total_conflicts(sud):
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

def is_solved(sud):
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

def random_complete_assignment(sud):
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

def ga_fitness(individual):
    """Fitness function: higher is better"""
    return -total_conflicts(individual)

def ga_tournament_select(population, fitnesses, k):
    """Tournament selection"""
    tournament_indices = random.sample(range(len(population)), k)
    best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
    return {'grid': population[best_idx]['grid'].copy(), 'givens_mask': population[best_idx]['givens_mask']}

def ga_crossover(p1, p2):
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

def ga_mutate(individual, rate, sud):
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

def solve_ga_single_run(sud, pop_size, crossover_rate, mutation_rate, tournament_k, max_generations, seed):
    """Single GA run with specific seed"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize population
    population = [random_complete_assignment(sud) for _ in range(pop_size)]
    
    for gen in range(max_generations):
        # Check if any individual is solved
        for individual in population:
            if total_conflicts(individual) == 0:
                return individual, gen, True
        
        # Calculate fitnesses
        fitnesses = [ga_fitness(x) for x in population]
        
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

class ReliableGASolver:
    """GA solver that tries multiple strategies to ensure success"""
    
    def __init__(self, puzzle_path: str):
        self.puzzle_path = puzzle_path
        self.sud = None
        self.load_puzzle()
    
    def load_puzzle(self):
        """Load the puzzle"""
        grid = read_puzzle_txt(self.puzzle_path)
        self.sud = sudoku_from_grid(grid)
        print(f"Loaded puzzle: {self.puzzle_path}")
        print(f"Initial conflicts: {total_conflicts(self.sud)}")
    
    def solve_with_multiple_seeds(self, max_attempts: int = 10, timeout_per_attempt: int = 30):
        """Try multiple seeds until success"""
        print(f"\n🔄 Strategy 1: Multiple Seeds (max {max_attempts} attempts)")
        
        # Optimized parameters from auto-tuning
        pop_size = 200
        crossover_rate = 0.77
        mutation_rate = 0.043
        tournament_k = 5
        max_generations = 1000
        
        for attempt in range(max_attempts):
            seed = random.randint(1, 10000)  # Random seed each time
            print(f"  Attempt {attempt + 1}: seed {seed}")
            
            start_time = time.time()
            solution, generations, success = solve_ga_single_run(
                self.sud, pop_size, crossover_rate, mutation_rate, 
                tournament_k, max_generations, seed
            )
            runtime = time.time() - start_time
            
            if success:
                print(f"  ✅ SUCCESS! Runtime: {runtime:.1f}s, Generations: {generations}")
                return solution, runtime, generations
            else:
                conflicts = total_conflicts(solution)
                print(f"  ❌ Failed. Conflicts: {conflicts}, Runtime: {runtime:.1f}s")
                
                # Check timeout
                if runtime > timeout_per_attempt:
                    print(f"  ⏰ Timeout after {timeout_per_attempt}s")
                    break
        
        print(f"  ❌ Failed after {max_attempts} attempts")
        return None, 0, 0
    
    def solve_with_adaptive_parameters(self, max_attempts: int = 5):
        """Try different parameter combinations"""
        print(f"\n🔧 Strategy 2: Adaptive Parameters (max {max_attempts} attempts)")
        
        # Different parameter sets to try
        param_sets = [
            # Original optimized
            (200, 0.77, 0.043, 5, 1000),
            # More aggressive
            (300, 0.85, 0.06, 7, 1500),
            # More conservative
            (150, 0.70, 0.03, 3, 800),
            # High diversity
            (250, 0.80, 0.08, 5, 1200),
            # Large population
            (400, 0.75, 0.05, 10, 2000)
        ]
        
        for attempt, (pop_size, crossover_rate, mutation_rate, tournament_k, max_generations) in enumerate(param_sets[:max_attempts]):
            print(f"  Attempt {attempt + 1}: pop={pop_size}, cross={crossover_rate:.2f}, mut={mutation_rate:.3f}, k={tournament_k}")
            
            # Try multiple seeds with this parameter set
            for seed_attempt in range(3):  # 3 seeds per parameter set
                seed = random.randint(1, 10000)
                start_time = time.time()
                
                solution, generations, success = solve_ga_single_run(
                    self.sud, pop_size, crossover_rate, mutation_rate, 
                    tournament_k, max_generations, seed
                )
                runtime = time.time() - start_time
                
                if success:
                    print(f"  ✅ SUCCESS! Runtime: {runtime:.1f}s, Generations: {generations}")
                    return solution, runtime, generations
                else:
                    conflicts = total_conflicts(solution)
                    print(f"    Seed {seed}: conflicts={conflicts}, time={runtime:.1f}s")
        
        print(f"  ❌ Failed with all parameter combinations")
        return None, 0, 0
    
    def solve_with_restart_strategy(self, max_attempts: int = 8):
        """Restart with different strategies"""
        print(f"\n🔄 Strategy 3: Restart Strategy (max {max_attempts} attempts)")
        
        strategies = [
            ("Multiple Seeds", self.solve_with_multiple_seeds),
            ("Adaptive Parameters", self.solve_with_adaptive_parameters)
        ]
        
        for attempt in range(max_attempts):
            strategy_name, strategy_func = strategies[attempt % len(strategies)]
            print(f"\n  Attempt {attempt + 1}: {strategy_name}")
            
            if strategy_name == "Multiple Seeds":
                result = strategy_func(max_attempts=5, timeout_per_attempt=20)
            else:
                result = strategy_func(max_attempts=3)
            
            solution, runtime, generations = result
            if solution is not None:
                return solution, runtime, generations
        
        print(f"  ❌ All strategies failed")
        return None, 0, 0
    
    def solve_reliable(self, strategy: str = "auto"):
        """Main reliable solving method"""
        print("🎯 Reliable GA Solver")
        print("="*50)
        
        if strategy == "auto":
            # Try multiple seeds first (fastest)
            solution, runtime, generations = self.solve_with_multiple_seeds(max_attempts=8)
            if solution is not None:
                return solution, runtime, generations
            
            # Try adaptive parameters
            solution, runtime, generations = self.solve_with_adaptive_parameters(max_attempts=3)
            if solution is not None:
                return solution, runtime, generations
            
            # Try restart strategy
            solution, runtime, generations = self.solve_with_restart_strategy(max_attempts=5)
            return solution, runtime, generations
        
        elif strategy == "seeds":
            return self.solve_with_multiple_seeds()
        elif strategy == "adaptive":
            return self.solve_with_adaptive_parameters()
        elif strategy == "restart":
            return self.solve_with_restart_strategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

def main():
    """Test the reliable solver"""
    puzzle_path = "puzzles/Easy-P1.txt"
    
    solver = ReliableGASolver(puzzle_path)
    
    print("Choose strategy:")
    print("1. Auto (try multiple strategies)")
    print("2. Multiple Seeds only")
    print("3. Adaptive Parameters only")
    
    choice = input("Enter choice (1-3): ").strip()
    
    strategies = {"1": "auto", "2": "seeds", "3": "adaptive"}
    strategy = strategies.get(choice, "auto")
    
    start_time = time.time()
    solution, runtime, generations = solver.solve_reliable(strategy)
    total_time = time.time() - start_time
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    if solution is not None:
        print(f"✅ SUCCESS!")
        print(f"Strategy: {strategy}")
        print(f"Runtime: {runtime:.1f}s")
        print(f"Total time: {total_time:.1f}s")
        print(f"Generations: {generations}")
        print(f"Solved: {is_solved(solution)}")
        print(f"Conflicts: {total_conflicts(solution)}")
    else:
        print(f"❌ FAILED")
        print(f"Total time: {total_time:.1f}s")
        print("Try running again or using a different strategy")

if __name__ == "__main__":
    main()
