#!/usr/bin/env python3
"""
Progress Checker for Auto GA Tuning

Quick script to check the current progress and best results
without interrupting the main tuning process.
"""

import json
import os
from datetime import datetime

def check_progress(results_file="auto_ga_results.json"):
    """Check current tuning progress"""
    
    if not os.path.exists(results_file):
        print(f"❌ Results file '{results_file}' not found!")
        print("Make sure the auto tuning script is running.")
        return
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        best_result = data.get('best_result')
        total_tests = data.get('total_tests', 0)
        successful_tests = data.get('successful_tests', 0)
        last_updated = data.get('last_updated', 'Unknown')
        puzzle_path = data.get('puzzle_path', 'Unknown')
        
        print("🔍 GA Tuning Progress Report")
        print("="*50)
        print(f"Puzzle: {puzzle_path}")
        print(f"Last updated: {last_updated}")
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success rate: {successful_tests/max(1,total_tests)*100:.1f}%")
        
        if best_result:
            print(f"\n🏆 Best Configuration Found:")
            print(f"   Population: {best_result['pop_size']}")
            print(f"   Crossover: {best_result['crossover_rate']:.2f}")
            print(f"   Mutation: {best_result['mutation_rate']:.3f}")
            print(f"   Tournament K: {best_result['tournament_k']}")
            print(f"   Runtime: {best_result['runtime']:.1f}s")
            print(f"   Decisions: {best_result['decisions']}")
            print(f"   Found at: {best_result.get('timestamp', 'Unknown')}")
        else:
            print(f"\n❌ No successful solutions found yet")
        
        # Show recent results
        if results:
            print(f"\n📊 Recent Results (last 5):")
            recent = results[-5:]
            for i, result in enumerate(recent, 1):
                status = "✅" if result['success'] else "❌"
                print(f"   {i}. {status} pop={result['pop_size']}, "
                      f"cross={result['crossover_rate']:.2f}, "
                      f"mut={result['mutation_rate']:.3f}, "
                      f"k={result['tournament_k']} "
                      f"({result['runtime']:.1f}s)")
        
        # Show top configurations by conflicts (if no successes)
        if not best_result and results:
            print(f"\n🔍 Best Failed Configurations (lowest conflicts):")
            failed_results = [r for r in results if not r['success']]
            failed_results.sort(key=lambda x: x['final_conflicts'])
            
            for i, result in enumerate(failed_results[:3], 1):
                print(f"   {i}. pop={result['pop_size']}, "
                      f"cross={result['crossover_rate']:.2f}, "
                      f"mut={result['mutation_rate']:.3f}, "
                      f"k={result['tournament_k']} "
                      f"(conflicts={result['final_conflicts']}, {result['runtime']:.1f}s)")
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")

if __name__ == "__main__":
    check_progress()
