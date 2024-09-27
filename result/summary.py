import os
import pandas as pd
from collections import defaultdict

def process_csv_files():
    max_rank = 0
    baseline = None
    records = 0
    evals = defaultdict(lambda: defaultdict(list))
    delta = defaultdict(lambda: defaultdict(list))
    count = defaultdict(lambda: defaultdict(int))
    
    # Process all CSV-like files in the current directory
    for filename in sorted(os.listdir('.')):
        if filename.endswith(".csv"):
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Prepare a list for cleaned data
            data = []
            seen = set()
            
            for line in lines:
                line = line.strip()

                # Skip comment or metadata lines (lines starting with '#')
                if line.startswith('#') or len(line) == 0:
                    continue

                # Split the line by commas to get fields
                parts = line.split(',')
                
                if len(parts) >= 4:
                    # Clean up and convert values as necessary
                    rank, rx, eval_val, baseline_val = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    
                    if rx == "baseline":
                        baseline = float(baseline_val)
                    else:
                        rank = int(rank)
                        eval_val = float(eval_val)
                        baseline_val = float(baseline_val)
                        
                        # Calculate delta if baseline is available
                        if baseline is not None:
                            delta_val = (baseline - baseline_val) / (baseline + 1e-30)
                            delta[rx][rank].append(delta_val)
                        
                        evals[rx][rank].append(eval_val)
                        count[rx][rank] += 1
                        max_rank = max(max_rank, rank)
                        records += 1

    return evals, delta, count, max_rank, records

def main():
    evals, delta, count, max_rank, records = process_csv_files()

    def mu(arr):
        if len(arr) == 0:
            return None
        return sum(arr) / len(arr)

    def sd(arr):
        if len(arr) < 2:
            return None
        return (max(arr) - min(arr)) / 2.56

    def print_ranks():
        print("RANK", end="")
        for rank in range(max_rank + 1):
            print(f", {rank:3}", end="")
        print("")

        for rx in count:
            print(f"{rx:10}", end="")
            for rank in range(max_rank + 1):
                if rank in count[rx]:
                    print(f", {count[rx][rank]:3}", end="")
                else:
                    print(",   ", end="")
            print("")

    def print_evaluations():
        print("\n#\n#EVALS\nRANK", end="")
        for rank in range(max_rank + 1):
            print(f", {rank:9}", end="")
        print("")

        for rx in evals:
            print(f"{rx:10}", end="")
            for rank in range(max_rank + 1):
                if rank in evals[rx]:
                    mean_val = mu(evals[rx][rank])
                    sd_val = sd(evals[rx][rank])
                    # Check if mean_val or sd_val is None, and print '-' if so
                    if mean_val is not None and sd_val is not None:
                        print(f", {mean_val:3.2f} ({sd_val:3.2f})", end="")
                    else:
                        print(",       -      ", end="")
                else:
                    print(",          ", end="")
            print("")


    def print_improvement():
        print("\n#\n#DELTAS\nRANK", end="")
        for rank in range(max_rank + 1):
            print(f", {rank:9}", end="")
        print("")

        for rx in delta:
            print(f"{rx:10}", end="")
            for rank in range(max_rank + 1):
                if rank in delta[rx]:
                    mean_delta = mu(delta[rx][rank])
                    sd_delta = sd(delta[rx][rank])
                    print(f", {mean_delta * 100:.1f}% ({sd_delta * 100:.1f}%)", end="")
                else:
                    print(",          ", end="")
            print("")

    # Print results
    print_ranks()
    print_evaluations()
    print_improvement()

if __name__ == "__main__":
    main()
