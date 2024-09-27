import stats
import sys, random, os
from ezr import the, DATA, csv, dot
import time
from math import exp, log, cos, sqrt, pi

def show(lst):
    return print(*[f"{word:6}" for word in lst], sep="\t")

def myfun(train_file):
    print(train_file)
    # Check if the file exists before proceeding
    if not os.path.isfile(train_file):
        print(f"Error: File {train_file} not found.")
        return
    
    try:
        # Print the file being opened for debugging
        print(f"Attempting to open: {train_file}")
        
        # Use the passed argument for the file path
        the.train = train_file  
        
        repeats = 20
        d = DATA().adds(csv(the.train))
        b4 = [d.chebyshev(row) for row in d.rows]
        somes = [stats.SOME(b4, f"asIs,{len(d.rows)}")]
        rnd = lambda z: z
        scoring_policies = [
            ('exploit', lambda B, R: B - R),
            ('explore', lambda B, R: (exp(B) + exp(R)) / (1E-30 + abs(exp(B) - exp(R))))]
        
        for what, how in scoring_policies:
            for the.Last in [0, 20, 30, 40]:
                for the.branch in [False, True]:
                    start = time.time()
                    result = []
                    runs = 0
                    for _ in range(repeats):
                        tmp = d.shuffle().activeLearning(score=how)
                        runs += len(tmp)
                        result += [rnd(d.chebyshev(tmp[0]))]

                    pre = f"{what}/b={the.branch}" if the.Last > 0 else "rrp"
                    tag = f"{pre},{int(runs/repeats)}"
                    print(tag, f": {(time.time() - start) / repeats:.2f} secs")
                    somes += [stats.SOME(result, tag)]
        
        pre = f"{what}/b={the.branch}" if the.Last > 0 else "rrp"
        tag = f"{pre},{int(runs/repeats)}"
        somes += [stats.SOME(result, tag)]
        stats.report(somes, 0.01)
    
    except FileNotFoundError:
        print(f"File {train_file} not found.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Not needed here, but good practice to always take care of seeds
random.seed(the.seed)

# Show header
show(["dim", "size", "xcols", "ycols", "rows", "file"])
show(["------"] * 6)

# Pass .csv files from command-line arguments to the function
[myfun(arg) for arg in sys.argv if arg.endswith(".csv")]
