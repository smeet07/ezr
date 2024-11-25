from copy import deepcopy
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from ezr import *
from enum import Enum

class DiscretizeTypes(Enum):
  EqualFrequencyBins = "quantile"
  KMeans = "kmeans"
        
class DiscretizationProcessor:
  def __init__(self, type: DiscretizeTypes, bins:int):
    self.type = type
    self.bins = bins

  def transform(self, dataset:DATA):
    new_data = deepcopy(dataset)
    sym_cols = [(i, col) for i, col in enumerate(new_data.cols.x) if isinstance(col,SYM)]
    num_cols = [(i, col) for i, col in enumerate(new_data.cols.x) if isinstance(col,NUM)]

    if len(num_cols) == 0:
        return new_data

    # Count unique values in each numeric column
    unique_counts = [len(set(r[col.at] for r in new_data.rows)) for _, col in num_cols]
    
    # Adjust number of bins for each column
    adjusted_bins = [min(self.bins, count) for count in unique_counts]

    # Use QuantileTransformer instead of KBinsDiscretizer for more flexibility
    from sklearn.preprocessing import QuantileTransformer

    x = np.array([[r[col.at] for _, col in num_cols] for r in new_data.rows])
    
    discretized_data = []
    for i, n_bins in enumerate(adjusted_bins):
        qt = QuantileTransformer(n_quantiles=n_bins, output_distribution='uniform')
        column_data = qt.fit_transform(x[:, i].reshape(-1, 1))
        discretized_data.append(np.floor(column_data * n_bins).astype(int))

    x = np.column_stack(discretized_data)

    # Rest of the method remains the same
    new_num_cols = []
    for i, (index, col) in enumerate(num_cols):
        new_col = SYM(at=col.at, txt=col.txt.lower())
        new_num_cols.append((index, new_col))
        for j, r in enumerate(new_data.rows):
            val = int(x[j][i])
            r[col.at] = val
            new_col.add(val)

    new_cols = [col for _, col in sorted(sym_cols + new_num_cols, key=lambda x: x[0])]

    for col_old, col_new in zip(new_data.cols.x, new_cols):
        index = new_data.cols.all.index(col_old)
        new_data.cols.all[index] = col_new

    new_data.cols.x = new_cols
    return new_data
  
def main():
  d = DATA().adds(csv(the.train))
  preprocessor = DiscretizationProcessor(DiscretizeTypes.EqualFrequencyBins, 5)
  new_data = preprocessor.transform(d)
  print([col.txt for col in new_data.cols.x])
  for col in new_data.cols.x:
    assert isinstance(col, SYM)
    
if __name__ == '__main__':
  main()