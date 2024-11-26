from ezr import *
from copy import deepcopy
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from modules.one_hot import OneHotPreprocessor
from modules.discretization import DiscretizationProcessor, DiscretizeTypes
import pandas as pd
        
class FeatureEliminationProcessor:
    def __init__(self, relevance, similarity, n_components=5):
        self.relevance = relevance
        self.similarity = similarity
        self.n_components = n_components
        
    def transform(self, dataset:DATA):
        data = deepcopy(dataset)
        columns_to_remove = []
        # sort columns by relevance
        columns = [(col, self.relevance.calculate(data, col)) for col in data.cols.x]
        # filter out columns with low relevance and sort by relevance
        columns = [col for col, _ in sorted(columns, key=lambda x: x[1], reverse=True)]
        i, j = 0, 1
        good_columns_count = 0
        while j < len(columns):
            if good_columns_count >= self.n_components or self.similarity.calculate(data, columns[i], columns[j]) >= self.similarity.threshold:
                columns_to_remove.append(columns[j])
            else:
                i = j
                good_columns_count += 1
            j += 1
        # modify the DATA object
        
        rows = np.array(data.rows)
        filterr = [col not in columns_to_remove for col in data.cols.all]
        data.rows = [[int(v) for v in row] for row in rows[:, filterr]]
        data.cols.all = [col for col in data.cols.all if col not in columns_to_remove]
        data.cols.x = [col for col in data.cols.x if col not in columns_to_remove]
        for i, col in enumerate(data.cols.all):
            col.at = i
        
            
        return DATA().adds([[col.txt for col in data.cols.all]] + data.rows)
            
    
class CosineSimilarity:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        
    def calculate(self, data, col1: COL, col2: COL):
        if not isinstance(col1, NUM) or not isinstance(col2, NUM):
            raise ValueError("Both columns should be of type NUM")
        x = np.array([row[col1.at] for row in data.rows])
        y = np.array([row[col2.at] for row in data.rows])
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
class MutualInformationSimilarity:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def calculate(self, data, col1: COL, col2: COL):
        if not isinstance(col1, SYM) or not isinstance(col2, SYM):
            raise ValueError("Both columns should be of type SYM")
        x = np.array([row[col1.at] for row in data.rows])
        y = np.array([row[col2.at] for row in data.rows])
        return normalized_mutual_info_score(x, y)
    
class VarianceRelevance:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def calculate(self, data, col: NUM):
        if not isinstance(col, NUM):
            raise ValueError("Column should be of type NUM")
        return col.mid()
    
class EntropyRelevance:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def calculate(self, data, col: SYM):
        if not isinstance(col, SYM):
            raise ValueError("Column should be of type SYM")
        return col.mid()
        
def main():
    d = DATA().adds(csv(the.train))
    d = OneHotPreprocessor().transform(d)
    preprocessor = FeatureEliminationProcessor(VarianceRelevance(), CosineSimilarity(0.85))
    new_data = preprocessor.transform(d)
    print([col.txt for col in new_data.cols.x])
    for col in new_data.cols.x:
        assert isinstance(col, NUM)
    for row in new_data.rows[:10]:
        print(row)
        
if __name__ == '__main__':
    main()