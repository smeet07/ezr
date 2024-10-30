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
        results.append((pipeline.n_components, the.Last, result, duration, len(d.cols.x), len(reduced_data.cols.x)))
    return results

class Pipeline:
    def __init__(self, name, preprocessors, n_components=None):
        self.preprocessors = preprocessors
        self.name = name
        self.n_components = n_components
        
    def transform(self, d):
        for preprocessor in self.preprocessors:
            d = preprocessor.transform(d)
        return d

original = Pipeline("Original", [])

def get_pipelines(d):
    pipelines = []
    for ratio in range(2, 9, 2):
        n_components = int(ratio * len(d.cols.x) / 10)
        sym_dim_red = Pipeline("Sym Dim Red", [
            DiscretizationProcessor(DiscretizeTypes.KMeans, 5),
            MCAProcessor(n_components)
        ], n_components)

        num_dim_red = Pipeline("Num Dim Red", [
            OneHotPreprocessor(),
            PCAProcessor(n_components=n_components)
        ], n_components)

        sym_feature_elim = Pipeline("Sym Feature Elim", [
            DiscretizationProcessor(DiscretizeTypes.KMeans, 5),
            FeatureEliminationProcessor(EntropyRelevance(0), MutualInformationSimilarity(0.7), n_components)
        ], n_components)

        num_feature_elim = Pipeline("Num Feature Elim", [
            OneHotPreprocessor(),
            FeatureEliminationProcessor(VarianceRelevance(0), CosineSimilarity(0.95), n_components)
        ], n_components)

        famd = Pipeline("FAMD", [ 
            FAMDProcessor(n_components)
        ], n_components)

        pipelines.extend([sym_feature_elim, original, sym_dim_red, num_dim_red, num_feature_elim, famd])

    return pipelines

import os
import pandas as pd

def main():
    datasets_path = "data/our_datasets"
    df = pd.DataFrame(columns=["dataset", "method", "n_components", "last", "mean", "std", "time", "xcols", "xcols_reduced", "rank"])
    index = 0
    for file in os.listdir(datasets_path):
        if not file.endswith(".csv"):
            continue
        print(f"Running {file}...")
        train_file = os.path.join(datasets_path, file)
            

        try:
            the.train = train_file
            d = DATA().adds(csv(the.train))

            print(f"rows: {len(d.rows)}")
            print(f"xcols: {len(d.cols.x)}")
            print(f"ycols: {len(d.cols.y)}\n")

            all_results = []
            pipelines = get_pipelines(d)
            for pipeline in pipelines:
                results = run_experiment(d, pipeline)
                print(f"Results for {pipeline.name}")
                all_results.extend([(pipeline.name, *result) for result in results])


            # Print results
            for method_name, n_components, last, result, duration, xcols, xcols_reduced in all_results:
                print(f"{method_name}, {n_components} Last={last}: mean={np.mean(result):.3f}, std={np.std(result):.3f}, time={duration:.2f} secs, xcols={xcols}, xcols_reduced={xcols_reduced}")
                df.loc[index] = [file, method_name, n_components, last, np.mean(result), np.std(result), duration, xcols, xcols_reduced, -1]
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