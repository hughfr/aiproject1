# Evil Puzzle GA Tuning Guide

## 🎯 **Updated Auto Tuner for Harder Puzzles**

The auto tuner has been significantly upgraded to handle evil puzzles and other difficult Sudoku puzzles!

## **Key Improvements**

### **1. Difficulty-Based Parameter Generation**
The tuner now automatically adjusts parameters based on puzzle difficulty:

| Difficulty | Population | Mutation Rate | Tournament K | Generations | Timeout |
|------------|------------|---------------|--------------|-------------|---------|
| **Easy**   | 100-400    | 0.01-0.12     | 3-10         | 500-1500    | 120s    |
| **Medium** | 200-600    | 0.03-0.15     | 5-20         | 1000-3000   | 300s    |
| **Hard**   | 300-800    | 0.03-0.15     | 5-20         | 2000-5000   | 600s    |
| **Evil**   | 300-800    | 0.03-0.15     | 5-20         | 2000-5000   | 600s    |

### **2. Puzzle Selection Interface**
The tuner now provides an interactive menu:
```
Choose puzzle difficulty:
1. Easy puzzles (P1-P4)
2. Medium puzzles (P1-P4)  
3. Hard puzzles (P1-P4)
4. Evil puzzles (P1-P4)
5. Custom puzzle
```

### **3. Increased Generation Limits**
- **Easy**: 500-1500 generations
- **Medium**: 1000-3000 generations  
- **Hard/Evil**: 2000-5000 generations

### **4. Longer Timeouts**
- **Easy**: 2 minutes per attempt
- **Medium**: 5 minutes per attempt
- **Hard/Evil**: 10 minutes per attempt

## **How to Use**

### **Option 1: Run the Batch File**
```bash
run_auto_tuning.bat
```
Then select option 4 for evil puzzles.

### **Option 2: Run Python Directly**
```bash
python auto_ga_tuning.py
```

### **Option 3: Test Specific Puzzle**
```bash
python auto_ga_tuning.py
# Choose option 5 and enter "Evil-P1.txt"
```

## **Expected Results for Evil Puzzles**

### **What to Expect:**
- **Much longer runtimes** (5-10 minutes per attempt)
- **Higher population sizes** (300-800 individuals)
- **More generations** (2000-5000 per attempt)
- **Lower success rates** initially
- **Better final solutions** when successful

### **Success Indicators:**
- **Conflicts decreasing** over time
- **Best conflicts < 10** is good progress
- **Best conflicts < 5** is very good
- **Best conflicts = 0** is success!

## **Notebook Updates**

Your notebook has also been updated:
- **GA_MAX_GENERATIONS**: Increased from 1000 to 2000
- **Reliable GA function**: Automatically retries with different seeds
- **Better parameter ranges**: Optimized for harder puzzles

## **Troubleshooting Evil Puzzles**

### **If Getting Stuck at 8+ Conflicts:**
1. **Run the auto tuner** on evil puzzles for several hours
2. **Use larger populations** (500-800)
3. **Try higher mutation rates** (0.08-0.15)
4. **Increase generations** to 3000-5000
5. **Use the reliable GA** with multiple seed attempts

### **Recommended Approach:**
1. **Start with auto tuner** on Evil-P1 for 2-4 hours
2. **Check results** for best parameters
3. **Update notebook** with best parameters found
4. **Test on multiple evil puzzles** to validate

## **Example Evil Puzzle Parameters**

Based on testing, good parameters for evil puzzles might be:
```python
GA_POP = 500              # Large population
GA_CROSSOVER_RATE = 0.85  # High crossover
GA_MUTATION_RATE = 0.08   # Higher mutation
GA_TOURNAMENT_K = 10      # Larger tournament
GA_MAX_GENERATIONS = 3000 # Many generations
```

## **Results Files**

Results are saved with puzzle-specific names:
- `auto_ga_results_Easy-P1.json`
- `auto_ga_results_Med-P1.json`
- `auto_ga_results_Hard-P1.json`
- `auto_ga_results_Evil-P1.json`

## **Tips for Success**

1. **Be patient** - Evil puzzles take much longer
2. **Run overnight** - Let the tuner run for 6+ hours
3. **Check progress** - Use the progress checker script
4. **Multiple attempts** - The reliable GA will try many seeds
5. **Parameter tuning** - Use auto tuner results to optimize

## **Next Steps**

1. **Run auto tuner** on Evil-P1 for several hours
2. **Check results** for best parameters
3. **Update your notebook** with optimal parameters
4. **Test on other evil puzzles** to validate
5. **Fine-tune** based on results

The updated system should significantly improve your success rate on evil puzzles! 🚀

