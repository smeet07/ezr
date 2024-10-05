import os
import re
from collections import defaultdict

def process_csv_files():
    max_rank = 0
    records = 0
    baseline = 1.0
    count = defaultdict(lambda: defaultdict(int))
    evals = defaultdict(lambda: defaultdict(list))
    delta = defaultdict(lambda: defaultdict(list))
    best_rank = defaultdict(lambda: float('inf'))

    for filename in sorted(os.listdir('.')):
        if filename.endswith('.csv'):
            with open(filename, 'r') as f:
                lines = f.readlines()
                data = []
                for line in lines:
                    line = re.sub(r'[ \t]', '', line)
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        data.append(parts)
                
                data.sort(key=lambda x: (int(x[0]), float(x[2])))
                seen = set()
                for parts in data:
                    if parts[1] not in seen:
                        rank = int(parts[0])
                        rx = parts[1]
                        eval_val = float(parts[2])
                        baseline_val = float(parts[3])

                        max_rank = max(max_rank, rank)
                        count[rx][rank] += 1
                        evals[rx][rank].append(eval_val)
                        best_rank[rx] = min(best_rank[rx], rank)

                        if rx == "asIs":
                            baseline = eval_val
                        else:
                            delta_val = (baseline - eval_val) / (baseline + 1e-30)
                            delta[rx][rank].append(delta_val)

                        seen.add(rx)
                        records += 1

    return max_rank, records, count, evals, delta, best_rank

def mu(a):
    if len(a) < 1:
        return 0
    return sum(a) / len(a)

def sd(a):
    if len(a) < 2:
        return 0
    a = sorted(a)
    n = max(1, int(len(a) / 10))
    return (a[-n] - a[n-1]) / 2.56

def per(x, records):
    x = int(0.5 + 100 * x / records)
    return "" if x < 1 else str(x)

def print_ranks(max_rank, records, count, best_rank):
    print("RANK", end="")
    for rank in range(max_rank + 1):
        print(f" {rank:>3}", end="")
    print()

    sorted_rx = sorted(count.keys(), key=lambda x: best_rank[x])
    for rx in sorted_rx:
        print(f"{rx:<10}", end="")
        for rank in range(max_rank + 1):
            print(f" {per(count[rx][rank], records):>3}", end="")
        print()

def print_evaluations(max_rank, evals, best_rank):
    print("\n#\n#EVALS\nRANK", end="")
    for rank in range(max_rank + 1):
        print(f" {rank:>9}", end="")
    print()

    sorted_rx = sorted(evals.keys(), key=lambda x: best_rank[x])
    for rx in sorted_rx:
        print(f"{rx:<10}", end="")
        for rank in range(max_rank + 1):
            if rank in evals[rx]:
                mean_val = mu(evals[rx][rank])
                sd_val = sd(evals[rx][rank])
                print(f" {int(0.5 + mean_val):3d} ({int(0.5 + sd_val):3d})", end="")
            else:
                print(" " * 9, end="")
        print()

def print_improvement(max_rank, delta, best_rank):
    print("\n#\n#DELTAS\nRANK", end="")
    for rank in range(max_rank + 1):
        print(f" {rank:>9}", end="")
    print()

    sorted_rx = sorted(delta.keys(), key=lambda x: best_rank[x])
    for rx in sorted_rx:
        print(f"{rx:<10}", end="")
        for rank in range(max_rank + 1):
            if rank in delta[rx]:
                mean_delta = mu(delta[rx][rank])
                sd_delta = sd(delta[rx][rank])
                print(f" {int(0.5 + 100 * mean_delta):3d} ({int(0.5 + 100 * sd_delta):3d})", end="")
            else:
                print(" " * 9, end="")
        print()

if __name__ == "__main__":
    max_rank, records, count, evals, delta, best_rank = process_csv_files()
    print_ranks(max_rank, records, count, best_rank)
    print_evaluations(max_rank, evals, best_rank)
    print_improvement(max_rank, delta, best_rank)
