# Sudoku Solver with Backtracking, Forward Checking, and AC-3

This project implements a comprehensive Sudoku solver using three different constraint satisfaction algorithms:

1. **Backtracking (BT)** - Basic backtracking with optional heuristics
2. **Backtracking + Forward Checking (BT-FC)** - Enhanced with forward checking
3. **Backtracking + AC-3 (BT-AC3)** - Enhanced with arc consistency

## Features

- **Three Solving Algorithms**: BT, BT-FC, and BT-AC3
- **Heuristics Support**: MRV (Minimum Remaining Values), DEG (Degree), and LCV (Least Constraining Value)
- **Decision Counting**: Tracks the number of variable assignments made during solving
- **Input/Output**: Reads puzzles from text files and writes solutions
- **Validation**: Ensures solutions are correct Sudoku puzzles

## File Structure

- `sudoku_solver.py` - Main solver implementation
- `test_sudoku.py` - Test script for all solvers
- `puzzle.txt` - Sample input puzzle
- `solved.txt` - Output file for solutions

## Usage

### Basic Usage

1. **Prepare your puzzle**: Create a text file with 9 lines, each containing 9 characters (digits 1-9 or '?' for empty cells)

2. **Configure the solver**: Edit the `CONFIG` dictionary in `sudoku_solver.py`:
   ```python
   CONFIG = {
       "input_path": "puzzle.txt",    # Input file path
       "output_path": "solved.txt",   # Output file path
       "solver": "bt",                # "bt", "bt-fc", or "bt-ac3"
       "use_mrv": True,               # Enable MRV heuristic
       "use_deg": True,               # Enable DEG heuristic
       "use_lcv": True,               # Enable LCV heuristic
   }
   ```

3. **Run the solver**:
   ```bash
   python sudoku_solver.py
   ```

### Testing

Run the test suite to verify all solvers work correctly:
```bash
python test_sudoku.py
```

## Puzzle Format

Input puzzles should be in the following format:
```
5?3???7??
6??195???
?98????6?
8???6??3?
4??8?3?1?
7???2??6?
?6????28?
???419??5
???8??79?
```

- Use digits 1-9 for given values
- Use '?' for empty cells
- Must have exactly 9 lines with 9 characters each

## Algorithms

### Backtracking (BT)
- Basic recursive backtracking
- Optional heuristics: MRV, DEG, LCV
- Incrementally assigns values and backtracks on conflicts

### Forward Checking (BT-FC)
- Maintains domains for each unassigned variable
- Prunes domains when values are assigned
- Fails fast when domains become empty

### AC-3 (BT-AC3)
- Applies arc consistency preprocessing
- Maintains arc-consistent domains throughout search
- More aggressive domain pruning

## Heuristics

- **MRV (Minimum Remaining Values)**: Choose variable with fewest legal values
- **DEG (Degree)**: Tie-breaker that chooses variable with most constraints
- **LCV (Least Constraining Value)**: Order values by how few constraints they impose

## Output

The solver provides:
- Solution written to output file
- Console report with:
  - Solver used
  - Number of decisions made
  - Whether puzzle was solved

## Example Output

```
Reading puzzle from puzzle.txt...
Puzzle loaded successfully.
Solving using bt...
Writing solution to solved.txt...
Solution written successfully.

Results:
Solver: bt
Decisions: 45
Solved: True
```

## Requirements

- Python 3.6+
- NumPy

## Implementation Details

The solver follows the pseudocode from the course materials:
- `BACKTRACKING-SEARCH` for basic backtracking
- `AC3` and `REMOVE-INCONSISTENT-VALUES` for arc consistency
- Forward checking as described in the CSP lectures

All algorithms maintain decision counters as required and implement the specified heuristics for variable and value selection.