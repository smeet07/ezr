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
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define lambda and epsilon for FOCUS function
LAMBDA = 0.25
EPSILON = 1E-30

def focus_score(B, R, t):
    try:
        # Avoid division by zero when B=1 by adjusting mt calculation
        mt = (exp(LAMBDA * t) - 1) / (exp(LAMBDA * (B - 1)) - 1) + 1 if B != 1 else 1
        score = ((B + 1) ** mt + (R + 1)) / (abs(B - R) + EPSILON)
        real_score = abs(score)  # Take the absolute value of the complex number if any
        logger.debug(f"FOCUS score calculation: B={B}, R={R}, t={t}, mt={mt}, score={real_score}")
        return real_score
    except Exception as e:
        logger.error(f"Error in FOCUS score calculation: {e}")
        return float('-inf')  # Return a very low score in case of error

scoring_policies = [
    ('exploit', lambda B, R: B - R),
    ('Random', lambda B, R: random.random()),
    ('FOCUS', lambda B, R: focus_score(B, R, the.iter)),
    ('explore', lambda B, R: (exp(B) + exp(R)) / (1E-30 + abs(exp(B) - exp(R))))
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
    n_components_list = []
    for ratio in range(2, 9, 2):
        original_n_components = int(ratio * len(d.cols.x) / 10)
        n_components_list.append(original_n_components)

    if n_components_list[0] >= 10:
        n_components_list = [5] + n_components_list

    for n_components in n_components_list:
        sym_dim_red = Pipeline(f"Symbolic Dimensionality Reduction ({n_components})", [
            DiscretizationProcessor(DiscretizeTypes.KMeans, 5),
            MCAProcessor(n_components)
        ], n_components)

        num_dim_red = Pipeline(f"Numeric Dimensionality Reduction ({n_components})", [
            OneHotPreprocessor(),
            PCAProcessor(n_components=n_components)
        ], n_components)

        sym_feature_elim = Pipeline(f"Symbolic Feature Elimation ({n_components})", [
            DiscretizationProcessor(DiscretizeTypes.KMeans, 5),
            FeatureEliminationProcessor(EntropyRelevance(0), MutualInformationSimilarity(0.7), n_components)
        ], n_components)

        num_feature_elim = Pipeline(f"Numeric Feature Elimination ({n_components})", [
            OneHotPreprocessor(),
            FeatureEliminationProcessor(VarianceRelevance(0), CosineSimilarity(0.95), n_components)
        ], n_components)

        famd = Pipeline(f"FAMD ({n_components})", [
            FAMDProcessor(n_components)
        ], n_components)

        pipelines.extend([sym_feature_elim, sym_dim_red, num_dim_red, num_feature_elim, famd])

    pipelines.append(original)
    return pipelines

def run_experiment(d, pipeline: 'Pipeline'):
    results = []
    the.Last = 20
    
    for policy_name, policy_func in scoring_policies:
        start = time.time()
        reduced_data = pipeline.transform(d)
        result = []
        for _ in range(the.Last):
            result.append(reduced_data.chebyshev(reduced_data.shuffle().activeLearning(score=policy_func)[0]))
        duration = (time.time() - start) / the.Last
        print(f"{pipeline.name} (Last={the.Last}, Policy={policy_name}): {duration:.2f} secs")
        results.append((pipeline.n_components, the.Last, policy_name, result, duration, len(d.cols.x), len(reduced_data.cols.x)))
    
    return results

def process_file(file, datasets_path):
    train_file = os.path.join(datasets_path, file)
    results = []
    
    try:
        the.train = train_file
        d = DATA().adds(csv(the.train))
        
        logger.info(f"Processing {file}...")
        logger.debug(f"rows: {len(d.rows)}, xcols: {len(d.cols.x)}, ycols: {len(d.cols.y)}")

        pipelines = get_pipelines(d)
        
        for pipeline in pipelines:
            try:
                pipeline_results = run_experiment(d, pipeline)
                results.extend([(file, pipeline.name, *result) for result in pipeline_results])
            except Exception as e:
                logger.error(f"An error occurred processing pipeline {pipeline.name} for {file}: {e}")

    except Exception as e:
        logger.error(f"An error occurred processing file {file}: {e}")

    return results

def main():
    datasets_path = "data/data2"
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, datasets_path): file 
                          for file in os.listdir(datasets_path) 
                          if file.endswith(".csv")}
        
        all_results_df_list = []

        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                results = future.result()
                
                df = pd.DataFrame(columns=["dataset", "method", "n_components", "last", "policy", "mean", "std", "time", "xcols", "xcols_reduced", "rank"])
                
                for result in results:
                    dataset, method_name, n_components, last, policy, result_values, duration, xcols, xcols_reduced = result
                    df_entry = {
                        "dataset": dataset,
                        "method": method_name,
                        "n_components": n_components,
                        "last": last,
                        "policy": policy,
                        "mean": np.mean(result_values),
                        "std": np.std(result_values),
                        "time": duration,
                        "xcols": xcols,
                        "xcols_reduced": xcols_reduced,
                        "rank": -1  # Initialize rank with -1
                    }
                    df = pd.concat([df, pd.DataFrame([df_entry])], ignore_index=True)

                # Generate SOME objects for stats report and update ranks
                somes = [stats.SOME(result[5], f"{result[1]},{result[3]},{result[4]}") for result in results]
                ranks = stats.report(somes, 0.01)

                for rank in ranks:
                    method_last_policy_strs= rank.txt.split(",")
                    if len(method_last_policy_strs) == 3:
                        method,last_str ,policy_str=method_last_policy_strs 
                        filterr =(df["method"] == method)&(df["last"] == int(last_str)) & (df["policy"] == policy_str)
                        df.loc[filterr,"rank"]=int(rank.rank)

                # Print output specific to each file before saving it to CSV
                print(f"\nResults for {file}:")
                print(df.to_string(index=False))

                all_results_df_list.append(df)

            except Exception as e:
                logger.error(f"An error occurred processing {file}: {e}")

    # Concatenate all DataFrames and save to a single CSV file
    final_df=pd.concat(all_results_df_list)
    final_df.to_csv("results.csv",index=False)

if __name__ == "__main__":
    random.seed(the.seed)
    main()