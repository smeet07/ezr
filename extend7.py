import sys
import random
import os
import time
import numpy as np
import pandas as pd
from math import exp
from ezr import the, DATA, csv
from modules.dimensionality import PCAProcessor, FAMDProcessor, MCAProcessor
from modules.discretization import DiscretizationProcessor, DiscretizeTypes
from modules.one_hot import OneHotPreprocessor
from modules.feature_elimination import FeatureEliminationProcessor, CosineSimilarity, MutualInformationSimilarity, VarianceRelevance, EntropyRelevance
import stats
import logging
from collections import namedtuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define lambda and epsilon for FOCUS function
LAMBDA = 0.25
EPSILON = 1E-30

def focus_score(B, R, t):
    try:
        mt = (exp(LAMBDA * t) - 1) / (exp(LAMBDA * (B - 1)) - 1) + 1 if B != 1 else 1
        score = ((B + 1) ** mt + (R + 1)) / (abs(B - R) + EPSILON)
        score = abs(score) # Ensure the score is not complex
        return max(1, min(2, score))  # Clamp the score between 1 and 2
    except OverflowError:
        return 1


Result = namedtuple("Result", ["dataset", "method", "n_components", "last", "policy", "mean", "std", "time", "xcols", "xcols_reduced", "rank", "all_results"])

Experiment = namedtuple("Experiment", ["data", "pipeline", "file", "policy", "last"])

Policy = namedtuple("Policy", ["name", "function"])

scoring_policies = [
    Policy('exploit', lambda B, R, _: B - R),
    Policy('Random', lambda B, R, t: random.random()),
    Policy('FOCUS', lambda B, R, t: focus_score(B, R, t)),
    Policy('explore', lambda B, R, t: (exp(B) + exp(R)) / (1E-30 + abs(exp(B) - exp(R))))
]

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
    # n_components_list = [int(ratio * len(d.cols.x) / 10) for ratio in range(2, 9, 2)]
    # if all([n_components > 10 for n_components in n_components_list]):
    #     n_components_list.append(5)
    
    n_components_list = [i for i in [5, 10, 20] if i < len(d.cols.x)]

    for n_components in n_components_list:
        sym_dim_red_kmeans = Pipeline(f"Symbolic Dimensionality Reduction ({n_components}) with Kmeans Discretization", [
            DiscretizationProcessor(DiscretizeTypes.KMeans, 5),
            MCAProcessor(n_components)
        ], n_components)
    
        sym_dim_red_efb = Pipeline(f"Symbolic Dimensionality Reduction ({n_components}) with EFB Discretization", [
            DiscretizationProcessor(DiscretizeTypes.EqualFrequencyBins, 5),
            MCAProcessor(n_components)
        ], n_components)

        num_dim_red = Pipeline(f"Numeric Dimensionality Reduction ({n_components})", [
            OneHotPreprocessor(),
            PCAProcessor(n_components=n_components)
        ], n_components)

        sym_feature_elim_kmeans = Pipeline(f"Symbolic Feature Elimation ({n_components}) with Kmeans Discretization", [
            DiscretizationProcessor(DiscretizeTypes.KMeans, 5),
            FeatureEliminationProcessor(EntropyRelevance(0), MutualInformationSimilarity(0.7), n_components)
        ], n_components)
        
        sym_feature_elim_efb = Pipeline(f"Symbolic Feature Elimation ({n_components}) with EFB Discretization", [
            DiscretizationProcessor(DiscretizeTypes.EqualFrequencyBins, 5),
            FeatureEliminationProcessor(EntropyRelevance(0), MutualInformationSimilarity(0.7), n_components)
        ], n_components)

        num_feature_elim = Pipeline(f"Numeric Feature Elimination ({n_components})", [
            OneHotPreprocessor(),
            FeatureEliminationProcessor(VarianceRelevance(0), CosineSimilarity(0.95), n_components)
        ], n_components)

        famd = Pipeline(f"FAMD ({n_components})", [
            FAMDProcessor(n_components)
        ], n_components)

        pipelines.extend([sym_dim_red_kmeans, sym_dim_red_efb, num_dim_red, sym_feature_elim_kmeans, sym_feature_elim_efb, num_feature_elim, famd])

    pipelines.append(original)
    return pipelines

def remove_dull_columns(d):
    cols_to_keep = [col for col in d.cols.x if col.div() > 0.1] + d.cols.y
    logger.info(f"Keeping {len(cols_to_keep)} columns out of {len(d.cols.all)}")
    
    rows = np.array(d.rows)
    filterr = [col in cols_to_keep for col in d.cols.all]
    rows = [[int(v) for v in row] for row in rows[:, filterr]]
    return DATA().adds([[col.txt for col in cols_to_keep]] + rows)

NR_RUNS = 20

def run_experiment(ex: Experiment):
    start = time.time()
    reduced_data = ex.pipeline.transform(ex.data)
    results = [reduced_data.chebyshev(reduced_data.shuffle().activeLearning(score=ex.policy.function)[0]) for _ in range(NR_RUNS)]
    duration = (time.time() - start) / NR_RUNS
    logger.info(f"{ex.pipeline.name} (Last={ex.last}, Policy={ex.policy.name}): {duration:.2f} secs")
    return Result(
        dataset=ex.file,
        method=ex.pipeline.name,
        n_components=ex.pipeline.n_components,
        last=ex.last,
        policy=ex.policy.name,
        mean=np.mean(results),
        std=np.std(results),
        time=duration,
        xcols=len(ex.data.cols.x),
        xcols_reduced=len(reduced_data.cols.x),
        rank=-1,
        all_results=results
    )

DATA_PATH = "data/data2"

def main():
    df = pd.DataFrame(columns=Result._fields)
    
    for data_path in [os.path.join(DATA_PATH, file) for file in os.listdir(DATA_PATH) if file.endswith(".csv")]:
        logger.info(f"Processing {data_path}...")
        data = DATA().adds(csv(data_path))
        data = remove_dull_columns(data)
        results = []
        
        for last in [20, 30, 40]: # we cannot run it parallel because it is global variable
            the.Last = last
            
            with ThreadPoolExecutor() as executor:
                futures = []
                
                for pipeline in get_pipelines(data):
                    
                    for policy in scoring_policies:
                        experiment = Experiment(data, pipeline, data_path, policy, last)
                        futures.append(executor.submit(run_experiment, experiment))
            
            for future in as_completed(futures):
                results.append(future.result())
                    
        df = pd.concat([df, pd.DataFrame([result._asdict() for result in results])], ignore_index=True)
        somes = [stats.SOME(result.all_results, f"{result.method},{result.last},{result.policy}") for result in results]
        ranks = stats.report(somes, 0.01)
        
        for rank in ranks:
            method, last_str, policy_str = rank.txt.split(",")
            filterr = (df["method"] == method) & (df["last"] == int(last_str)) & (df["policy"] == policy_str)
            df.loc[filterr, "rank"] = int(rank.rank)
            
        df.to_csv("results.csv", index=False)
        logger.info(f"Results for {data_path} saved to results.csv")

if __name__ == "__main__":
    random.seed(the.seed)
    main()