import sys
import random
import os
import time
import numpy as np
from ezr import the, DATA, csv
from modules.dimensionality import PCAProcessor, FAMDProcessor, MCAProcessor
from modules.discretization import DiscretizationProcessor, DiscretizeTypes
from modules.one_hot import OneHotPreprocessor
from modules.feature_elimination import FeatureEliminationProcessor, CosineSimilarity, MutualInformationSimilarity, VarianceRelevance, EntropyRelevance
import stats

def run_experiment(d,pipeline:'Pipeline'):
    results = []
    for the.Last in [20, 30, 40]:
        start = time.time()
        reduced_data = pipeline.transform(d)
        result = [reduced_data.chebyshev(reduced_data.shuffle().activeLearning()[0]) for _ in range(20)]
        duration = (time.time() - start) / 20
        print(f"{pipeline.name} (Last={the.Last}): {duration:.2f} secs")
        results.append((the.Last, result, duration, len(d.cols.x), len(reduced_data.cols.x)))
    return results

class Pipeline:
    def __init__(self, name, preprocessors):
        self.preprocessors = preprocessors
        self.name = name
        
    def transform(self, d):
        for preprocessor in self.preprocessors:
            d = preprocessor.transform(d)
        return d

original = Pipeline("Original", [])

sym_dim_red = Pipeline("Sym Dim Red", [
    DiscretizationProcessor(DiscretizeTypes.KMeans, 10),
    MCAProcessor(5)
])

num_dim_red = Pipeline("Num Dim Red", [
    OneHotPreprocessor(),
    PCAProcessor(n_components=10)
])

sym_feature_elim = Pipeline("Sym Feature Elim", [
    DiscretizationProcessor(DiscretizeTypes.KMeans, 10),
    FeatureEliminationProcessor(EntropyRelevance(0), MutualInformationSimilarity(0.7), 10)
])

num_feature_elim = Pipeline("Num Feature Elim", [
    OneHotPreprocessor(),
    FeatureEliminationProcessor(VarianceRelevance(0), CosineSimilarity(0.7), 10)
])

famd = Pipeline("FAMD", [ 
    FAMDProcessor(10)
])

import os
import pandas as pd

def main():
    datasets_path = "data/our_datasets"
    df = pd.DataFrame(columns=["dataset", "method", "last", "mean", "std", "time", "xcols", "xcols_reduced", "rank"])
    index = 0
    for file in os.listdir(datasets_path):
        if not file.endswith(".csv"):
            continue
        print(f"Running {file}...")
        train_file = os.path.join(datasets_path, file)
            
        pipelines = [sym_feature_elim, original, sym_dim_red, num_dim_red, num_feature_elim]

        try:
            the.train = train_file
            d = DATA().adds(csv(the.train))

            print(f"rows: {len(d.rows)}")
            print(f"xcols: {len(d.cols.x)}")
            print(f"ycols: {len(d.cols.y)}\n")

            all_results = []
            for pipeline in pipelines:
                results = run_experiment(d, pipeline)
                all_results.extend([(pipeline.name, *result) for result in results])

            # Print results
            for method_name, last, result, duration, xcols, xcols_reduced in all_results:
                print(f"{method_name}, Last={last}: mean={np.mean(result):.3f}, std={np.std(result):.3f}, time={duration:.2f} secs, xcols={xcols}, xcols_reduced={xcols_reduced}")
                df.loc[index] = [file, method_name, last, np.mean(result), np.std(result), duration, xcols, xcols_reduced, -1]
                index += 1

            # Generate SOME objects for stats report
            somes = [stats.SOME(result, f"{method_name},{last}") for method_name, last, result, _, _, _ in all_results]
            ranks = stats.report(somes, 0.01)
            for rank in ranks:
                method, last = rank.txt.split(",")
                filterr = (df["method"] == method) & (df["last"] == int(last)) & (df['dataset'] == file)
                df.loc[filterr, "rank"] = int(rank.rank)
            df.to_csv("results.csv", index=False)
            

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    random.seed(the.seed)
    main()