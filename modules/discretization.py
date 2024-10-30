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
    
    # discretize numeric columns
    discretizer = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy=self.type.value)
    x = np.array([[r[col.at] for _, col in num_cols] for r in new_data.rows])
    x = discretizer.fit_transform(x)
    
    # modify the DATA object based on new valies
    new_num_cols = []
    for i, (index, col) in enumerate(num_cols):
      new_col = SYM(at=col.at, txt=col.txt.lower()) # create a new SYM column
      new_num_cols.append((index, new_col))
      for j, r in enumerate(new_data.rows):
        val = int(x[j][i])
        r[col.at] = val # update the value in the row
        new_col.add(val) # update stats in the column

    new_cols = [col for _, col in sorted(sym_cols + new_num_cols, key=lambda x: x[0])]
    # update the column list all in the DATA object
    for col_old, col_new in zip(new_data.cols.x, new_cols):
      index = new_data.cols.all.index(col_old)
      new_data.cols.all[index] = col_new
    # update the column list x in the DATA object
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