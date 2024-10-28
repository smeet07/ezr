import sys
import random
import os
import time
import numpy as np
from ezr import the, DATA, csv
from dimensionality import PCAProcessor, FAMDProcessor
import stats

def run_experiment(d, method_name, reduction_func=None):
    results = []
    for the.Last in [20, 30, 40]:
        start = time.time()
        if reduction_func:
            reduced_data = reduction_func(d)
            result = [reduced_data.chebyshev(reduced_data.shuffle().activeLearning()[0]) for _ in range(20)]
        else:
            result = [d.chebyshev(d.shuffle().activeLearning()[0]) for _ in range(20)]
        duration = (time.time() - start) / 20
        print(f"{method_name} (Last={the.Last}): {duration:.2f} secs")
        results.append((the.Last, result, duration))
    return results

def mca_symbolic(d):
    famd = FAMDProcessor(d, n_components=3)
    return famd.fit_transform()

def famd_both(d):
    famd = FAMDProcessor(d, n_components=3)
    return famd.fit_transform()

def pca_numeric(d):
    pca = PCAProcessor(d, n_components=2, cat_method='none')
    return pca.fit_transform()

def main(train_file):
    print(train_file)
    if not os.path.isfile(train_file):
        print(f"Error: File {train_file} not found.")
        return

    try:
        the.train = train_file
        d = DATA().adds(csv(the.train))

        print(f"rows: {len(d.rows)}")
        print(f"xcols: {len(d.cols.x)}")
        print(f"ycols: {len(d.cols.y)}\n")

        reduction_methods = [
            ("Original", None),
            ("MCA Symbolic", mca_symbolic),
            ("FAMD Both", famd_both),
            ("PCA Numeric", pca_numeric)
        ]

        all_results = []
        for method_name, reduction_func in reduction_methods:
            results = run_experiment(d, method_name, reduction_func)
            all_results.extend([(method_name, *result) for result in results])

        # Print results
        for method_name, last, result, duration in all_results:
            print(f"{method_name}, Last={last}: mean={np.mean(result):.3f}, std={np.std(result):.3f}, time={duration:.2f} secs")

        # Generate SOME objects for stats report
        somes = [stats.SOME(result, f"{method_name},{last}") for method_name, last, result, _ in all_results]
        stats.report(somes, 0.01)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    random.seed(the.seed)
    [main(arg) for arg in sys.argv if arg.endswith(".csv")]