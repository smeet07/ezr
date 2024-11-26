from sklearn.preprocessing import OneHotEncoder
from ezr import *
import numpy as np

class OneHotPreprocessor:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)
    
    def transform(self, data: DATA) -> DATA:
        sym_cols = [col for i, col in enumerate(data.cols.x) if isinstance(col,SYM)]
        num_cols = [col for i, col in enumerate(data.cols.x) if isinstance(col,NUM)]
        y_cols = [col for i, col in enumerate(data.cols.y)]
        
        if len(sym_cols) == 0:
            return data
        
        cat_data = np.array([[r[col.at] for col in sym_cols] for r in data.rows])
        # if there are only 1 and 0, do not use OneHotEncoder
        if np.all(np.logical_or(cat_data == 0, cat_data == 1)):
            encoded = cat_data
        else:
            encoded = self.encoder.fit_transform(cat_data)
        
        index = 0
        new_cols = []
        new_cols_values = []
        for col in num_cols+y_cols:
            new_cols.append(NUM(at=index, txt=col.txt))
            new_cols_values.append([r[col.at] for r in data.rows])
            index += 1
        for i, col in enumerate(encoded.T):
            new_cols.append(SYM(at=index, txt=f'OneHot_{i}'))
            new_cols_values.append(col)
            index += 1
            
        rows = []
        for i in range(len(data.rows)):
            row = []
            for col_values in new_cols_values:
                row.append(col_values[i])
            rows.append(row)
            
        new_data = DATA().adds([[col.txt for col in new_cols]] + rows)
        assert all([isinstance(col, NUM) for col in new_data.cols.x])
        return new_data
        
        
def main():
    d = DATA().adds(csv(the.train))
    preprocessor = OneHotPreprocessor()
    new_data = preprocessor.transform(d)
    print([col.txt for col in new_data.cols.x])
    for row in new_data.rows:
        assert len(row) == len(new_data.cols.all)
    
if __name__ == '__main__':
    main()