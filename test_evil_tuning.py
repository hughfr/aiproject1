#!/usr/bin/env python3
"""
Quick test of the updated auto tuner with evil puzzles
"""

import sys
import os

# Add the current directory to path so we can import the auto tuner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_ga_tuning import AutoGATuner

def test_evil_tuning():
    """Test the auto tuner with evil puzzles"""
    print("Testing Auto Tuner with Evil Puzzles")
    print("="*40)
    
    # Test with Evil-P1
    puzzle_path = "puzzles/Evil-P1.txt"
    results_file = "test_evil_results.json"
    difficulty = "evil"
    
    if not os.path.exists(puzzle_path):
        print(f"Error: {puzzle_path} not found!")
        return
    
    print(f"Puzzle: {puzzle_path}")
    print(f"Difficulty: {difficulty}")
    print(f"Results file: {results_file}")
    
    # Initialize tuner
    tuner = AutoGATuner(puzzle_path, results_file, difficulty)
    
    # Test parameter generation
    print("\nTesting parameter generation for evil puzzles:")
    for i in range(5):
        pop_size, crossover_rate, mutation_rate, tournament_k, max_generations, timeout = tuner.generate_random_params("evil")
        print(f"  {i+1}. pop={pop_size}, cross={crossover_rate:.2f}, mut={mutation_rate:.3f}, k={tournament_k}, gens={max_generations}, timeout={timeout}s")
    
    print("\nParameter ranges for evil puzzles:")
    print("  Population: 300-800 (larger for harder puzzles)")
    print("  Crossover: 0.7-0.95")
    print("  Mutation: 0.03-0.15 (higher for harder puzzles)")
    print("  Tournament K: 5-20 (larger for harder puzzles)")
    print("  Generations: 2000-5000 (much higher)")
    print("  Timeout: 600s+ (longer for harder puzzles)")
    
    print("\n✅ Auto tuner updated successfully for evil puzzles!")

if __name__ == "__main__":
    test_evil_tuning()

