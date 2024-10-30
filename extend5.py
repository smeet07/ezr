import sys
import random
import os
import time
import numpy as np
from ezr import the, DATA, csv as ezr_csv
from dimensionality import PCAProcessor, FAMDProcessor
import stats
import csv

def calculate_n_components(original_dims, proportion):
    return max(1, int(original_dims * proportion))

def run_experiment(d, method_name, reduction_func=None, dim_proportion=1.0):
    results = []
    n_components = calculate_n_components(len(d.cols.x), dim_proportion)
    for the.Last in [20, 30, 40]:
        start = time.time()
        if reduction_func:
            reduced_data = reduction_func(d, n_components)
            result = [reduced_data.chebyshev(reduced_data.shuffle().activeLearning()[0]) for _ in range(20)]
        else:
            result = [d.chebyshev(d.shuffle().activeLearning()[0]) for _ in range(20)]
        duration = (time.time() - start) / 20
        print(f"{method_name} (Last={the.Last}, Dim={dim_proportion:.1f}): {duration:.2f} secs")
        results.append((the.Last, result, duration))
    return results

def mca_symbolic(d, n_components):
    famd = FAMDProcessor(d, n_components=n_components)
    return famd.fit_transform()

def famd_both(d, n_components):
    famd = FAMDProcessor(d, n_components=n_components)
    return famd.fit_transform()

def pca_numeric(d, n_components):
    pca = PCAProcessor(d, n_components=n_components, cat_method='none')
    return pca.fit_transform()

def main(train_file):
    print(train_file)
    if not os.path.isfile(train_file):
        print(f"Error: File {train_file} not found.")
        return

    try:
        the.train = train_file
        d = DATA().adds(ezr_csv(the.train))

        print(f"rows: {len(d.rows)}")
        print(f"xcols: {len(d.cols.x)}")
        print(f"ycols: {len(d.cols.y)}\n")

        reduction_methods = [
            ("Original", None),
            ("MCA Symbolic", mca_symbolic),
            ("FAMD Both", famd_both),
            ("PCA Numeric", pca_numeric)
        ]

        dim_proportions = [0.2, 0.4, 0.6, 0.8, 1.0]

        all_results = []
        for method_name, reduction_func in reduction_methods:
            for dim_prop in dim_proportions:
                results = run_experiment(d, method_name, reduction_func, dim_prop)
                all_results.extend([(method_name, dim_prop, *result) for result in results])

        # Print results
        for method_name, dim_prop, last, result, duration in all_results:
            print(f"{method_name}, Dim={dim_prop:.1f}, Last={last}: mean={np.mean(result):.3f}, std={np.std(result):.3f}, time={duration:.2f} secs")

        # Generate SOME objects for stats report
        somes = [stats.SOME(result, f"{method_name},Dim={dim_prop:.1f},Last={last}") for method_name, dim_prop, last, result, _ in all_results]
        ranked_somes = stats.sk(somes, 0.01)

        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

        # Write results to CSV
        output_file = os.path.join('results', os.path.basename(train_file).replace('.csv', '_results.csv'))
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Rank', 'Method', 'Dim', 'Last', 'Mean', 'Std'])
            for some in ranked_somes:
                method, dim, last = some.txt.split(',')
                writer.writerow([some.rank, method, dim, last, some.mid(), some.div()])

        print(f"Results written to {output_file}")

        # Print final rankings
        stats.report(ranked_somes, 0.01)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    random.seed(the.seed)
    [main(arg) for arg in sys.argv if arg.endswith(".csv")]