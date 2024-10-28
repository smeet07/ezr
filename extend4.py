import sys
import random
import os
import time
import numpy as np
from ezr import the, DATA, csv, dot
from dimensionality import PCAProcessor, FAMDProcessor
import stats

def show(lst):
    return print(*[f"{word:6}" for word in lst], sep="\t")

def run_experiment(d, method_name, reduction_func=None):
    results = []
    for the.Last in [20, 30, 40]:
        start = time.time()
        if reduction_func:
            try:
                reduced_data = reduction_func(d)
                print(f"Original data: rows={len(d.rows)}, x_cols={len(d.cols.x)}, y_cols={len(d.cols.y)}")
                print(f"Reduced data: rows={len(reduced_data.rows)}, x_cols={len(reduced_data.cols.x)}, y_cols={len(reduced_data.cols.y)}")
                result = [d.chebyshev(reduced_data.shuffle().activeLearning()[0]) for _ in range(20)]
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                print(f"First row of reduced data: {reduced_data.rows[0] if reduced_data.rows else 'No rows'}")
                result = []
        else:
            result = [d.chebyshev(d.shuffle().activeLearning()[0]) for _ in range(20)]
        duration = (time.time() - start) / 20
        print(f"{method_name} (Last={the.Last}): {duration:.2f} secs")
        results.append((the.Last, result, duration))
    return method_name, results

def compare_results(*method_results):
    for method_name, results in method_results:
        print(f"\nResults for {method_name}:")
        for last, result, duration in results:
            print(f"Last={last}: mean={np.mean(result):.3f}, std={np.std(result):.3f}, time={duration:.2f} secs")

def mca_symbolic(d):
    famd = FAMDProcessor(d, n_components=2)
    return famd.fit_transform()

def famd_both(d):
    famd = FAMDProcessor(d, n_components=2)
    return famd.fit_transform()

def pca_numeric(d):
    pca = PCAProcessor(d, n_components=2, cat_method='none')
    return pca.fit_transform()

def myfun(train_file):
    print(train_file)
    
    if not os.path.isfile(train_file):
        print(f"Error: File {train_file} not found.")
        return

    try:
        the.train = train_file
        repeats = 20
        d = DATA().adds(csv(the.train))
        
        b4 = [d.chebyshev(row) for row in d.rows]
        asIs = max(b4)
        div = min(b4)
        
        print(f"asIs: {asIs:.3f}")
        print(f"div : {div:.3f}")
        print(f"rows: {len(d.rows)}")
        print(f"xcols: {len(d.cols.x)}")
        print(f"ycols: {len(d.cols.y)}\n")
        
        somes = [stats.SOME(b4, f"asIs,{len(d.rows)}")]
        
        reduction_methods = [
            ("Original", None),
            ("MCA Symbolic", mca_symbolic),
            ("FAMD Both", famd_both),
            ("PCA Numeric", pca_numeric)
        ]
        
        for what, how in reduction_methods:
            method_name, results = run_experiment(d, what, how)
            for the_last, result, duration in results:
                if result:  # Only process if result is not empty
                    tag = f"{what},{the_last},{len(result)}"
                    print(f"{tag} : {duration:.2f} secs")
                    somes += [stats.SOME(result, tag)]
                else:
                    print(f"No results for {what}, Last={the_last}")
        
        stats.report(somes, 0.01)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    

# Main execution
if __name__ == "__main__":
    random.seed(the.seed)
    [myfun(arg) for arg in sys.argv if arg.endswith(".csv")]