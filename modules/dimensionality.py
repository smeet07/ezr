from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import stats
import sys, random
import ezr
from ezr import the, DATA, NUM, SYM, COL, csv, dot, adds
from prince import FAMD, MCA
import pandas as pd
from modules.one_hot import OneHotPreprocessor
from modules.discretization import DiscretizationProcessor, DiscretizeTypes

from sklearn.decomposition import PCA
from ezr import *
from copy import deepcopy
import numpy as np

from prince import MCA
from ezr import *
import numpy as np
import pandas as pd

class PCAProcessor:
    def __init__(self, n_components=None):
        self.pca = PCA(n_components=n_components)
        
    def transform(self, data: DATA) -> DATA:
        # Check that all columns are numeric
        data = deepcopy(data)
        num_cols = [col for col in data.cols.x if isinstance(col, NUM)]
        sym_cols = [col for col in data.cols.x if isinstance(col, SYM)]
        y_cols = data.cols.y
        
        assert len(sym_cols) == 0, "All columns must be numeric for PCA"
        assert len(num_cols) > 0, "No numeric columns to apply PCA"
            
        # Prepare the data matrix X
        X = np.array([[r[col.at] for col in num_cols] for r in data.rows])
        # Apply PCA
        X_pca = self.pca.fit_transform(X)
        
        # Create new columns for the PCA components
        new_cols = []
        new_cols_values = []
        index = 0
        for i in range(X_pca.shape[1]):
            new_cols.append(NUM(at=index, txt=f'PC_{i+1}'))
            new_cols_values.append([float(val) for val in X_pca[:, i]])
            index += 1
        
        # Include the y columns (if any)
        for col in y_cols:
            new_cols.append(col)
            new_cols_values.append([r[col.at] for r in data.rows])
            col.at = index  # Update the index
            index += 1
        
        # Build new rows
        rows = []
        for i in range(len(data.rows)):
            row = []
            for col_values in new_cols_values:
                row.append(col_values[i])
            rows.append(row)
        
        # Create new DATA object
        new_data = DATA().adds([[col.txt for col in new_cols]] + rows)
        # Confirm that the x columns are all NUM
        assert all([isinstance(col, NUM) for col in new_data.cols.x])
        return new_data

class MCAProcessor:
    def __init__(self, n_components=None):
        self.mca = MCA(n_components=n_components)
        
    def transform(self, data: DATA) -> DATA:
        # Check that all columns are categorical (SYM)
        data = deepcopy(data)
        sym_cols = [col for col in data.cols.x if isinstance(col, SYM)]
        num_cols = [col for col in data.cols.x if isinstance(col, NUM)]
        y_cols = data.cols.y
        
        assert len(sym_cols) > 0, "No categorical columns to apply MCA"
        assert len(num_cols) == 0, "All columns must be categorical (SYM) for MCA"
            
        # Prepare the data matrix X
        X = [[r[col.at] for col in sym_cols] for r in data.rows]
        X_df = pd.DataFrame(X, columns=[col.txt for col in sym_cols])
        
        # Apply MCA
        X_mca = self.mca.fit_transform(X_df)
        
        # Create new columns for the MCA components
        new_cols = []
        new_cols_values = []
        index = 0
        for i in range(X_mca.shape[1]):
            new_cols.append(NUM(at=index, txt=f'MCA_{i+1}'))
            new_cols_values.append([float(val) for val in X_mca.iloc[:, i].values])
            index += 1
        
        # Include the y columns (if any)
        for col in y_cols:
            new_cols.append(col)
            new_cols_values.append([r[col.at] for r in data.rows])
            col.at = index  # Update the index
            index += 1
        
        # Build new rows
        rows = []
        for i in range(len(data.rows)):
            row = []
            for col_values in new_cols_values:
                row.append(col_values[i])
            rows.append(row)
        
        # Create new DATA object
        new_data = DATA().adds([[col.txt for col in new_cols]] + rows)
        # Confirm that the x columns are all NUM
        assert all([isinstance(col, NUM) for col in new_data.cols.x])
        return new_data
    
from prince import FAMD
from ezr import *
import numpy as np
import pandas as pd

class FAMDProcessor:
    def __init__(self, n_components=None):
        self.famd = FAMD(n_components=n_components)
        
    def transform(self, data: DATA) -> DATA:
        data = deepcopy(data)
        # Separate numeric and categorical columns
        num_cols = [col for col in data.cols.x if isinstance(col, NUM)]
        sym_cols = [col for col in data.cols.x if isinstance(col, SYM)]
        y_cols = data.cols.y
        
        if len(num_cols) == 0:
            return data
        if len(sym_cols) == 0:
            return data
        
        # Prepare the data matrix X
        X = []
        for r in data.rows:
            row = {}
            for col in num_cols:
                row[col.txt] = float(r[col.at])
            for col in sym_cols:
                row[col.txt] = str(r[col.at])
            X.append(row)
        X_df = pd.DataFrame(X)
        
        # Apply FAMD
        X_famd = self.famd.fit_transform(X_df)
        
        # Create new columns for the FAMD components
        new_cols = []
        new_cols_values = []
        index = 0
        for i in range(X_famd.shape[1]):
            new_cols.append(NUM(at=index, txt=f'FAMD_{i+1}'))
            new_cols_values.append([float(val) for val in X_famd.iloc[:, i].values])
            index += 1
        
        # Include the y columns (if any)
        for col in y_cols:
            new_cols.append(col)
            new_cols_values.append([r[col.at] for r in data.rows])
            col.at = index  # Update the index
            index += 1
        
        # Build new rows
        rows = []
        for i in range(len(data.rows)):
            row = []
            for col_values in new_cols_values:
                row.append(col_values[i])
            rows.append(row)
        
        # Create new DATA object
        new_data = DATA().adds([[col.txt for col in new_cols]] + rows)
        # Confirm that the x columns are all NUM
        assert all([isinstance(col, NUM) for col in new_data.cols.x])
        return new_data
    
def main():
    d = DATA().adds(csv('data/our_datasets/FFM-125-25-0.50-SAT-1.csv'))
    preprocessor = FAMDProcessor(n_components=2)
    # new_data = DiscretizationProcessor(DiscretizeTypes.KMeans, 5).transform(d)
    new_data = preprocessor.transform(d)
    print([col.txt for col in new_data.cols.all])
    assert all([isinstance(col, NUM) for col in new_data.cols.x])
    assert len(new_data.cols.x) == 2
    
    for row in new_data.rows[:5]:
        print(row)
    
if __name__ == '__main__':
    main()


# class PCAProcessor:
#     def __init__(self, data: DATA, n_components: int, cat_method: str = 'none'):
#         self.data = data
#         self.n_components = n_components
#         self.cat_method = cat_method
#         self.pca = PCA(n_components=n_components)
#         self.encoder = None
#         self.num_cols = [col.txt for col in self.data.cols.x if isinstance(col, NUM)]
#         self.cat_cols = [col.txt for col in self.data.cols.x if isinstance(col, SYM)]
#         self.new_cols = []

#     def fit_transform(self) -> DATA:
#         # Extract and preprocess numeric and categorical data
#         numeric_data = self._extract_numeric_data()
#         cat_data = self._extract_categorical_data()
        
#         # Combine numeric and categorical data
#         combined_data = self._combine_data(numeric_data, cat_data)
    
#         if self.cat_method == 'none':
#             n_samples, n_features = numeric_data.shape
#             print("number of numeric columns: ", n_features)
#             pca_result = cat_data
#             if n_features>0 and n_samples > 0:
#                 pca_result= self.pca.fit_transform(numeric_data)
#                 pca_result = self._combine_data(pca_result, cat_data)
#         else:
#             n_samples, n_features = combined_data.shape
#             print("number of combined columns: ", n_features, "after ", self.cat_method)
#             if n_features>0 and n_samples > 0:
#                 pca_result= self.pca.fit_transform(combined_data)

#         pca_result_samples, pca_result_features = pca_result.shape   
#         print("number of pca columns: ", pca_result_features)    
#         # Create new DATA object with the PCA result
#         return DATA().adds(self._create_new_data_object(pca_result))

#     def _extract_numeric_data(self) -> np.ndarray:
#         return np.array([[row[col.at] for col in self.data.cols.x if isinstance(col, NUM)] for row in self.data.rows])

#     def _extract_categorical_data(self) -> np.ndarray:
#         return np.array([[row[col.at] for col in self.data.cols.x if isinstance(col, SYM)] for row in self.data.rows])

#     def _one_hot_encode(self, cat_data) -> np.ndarray:
#         self.encoder = OneHotEncoder(sparse_output=False)
#         encoded=self.encoder.fit_transform(cat_data)
#         return encoded

#     def _label_encode(self,cat_data) -> np.ndarray:
#         self.encoder = LabelEncoder()
#         return np.array([self.encoder.fit_transform(col) for col in cat_data.T]).T

#     def _combine_data(self, numeric_data: np.ndarray, cat_data: np.ndarray) -> np.ndarray:
#         if numeric_data.size == 0:
#             return cat_data
#         if cat_data.size == 0:
#             return numeric_data
#         return np.hstack((numeric_data, cat_data))

#     def _create_new_data_object(self, combined_data: np.ndarray) -> DATA:
#         y_cols = [col.txt for col in self.data.cols.y]
#         if self.cat_method == 'none':
#             self.new_cols = ['Pca'+str(i) for i in range(min(self.n_components, len(self.num_cols)))]
#             self.new_cols += self.cat_cols
#         else:
#             self.new_cols = ['Pca'+str(i) for i in range(self.n_components)]
#         self.new_cols += y_cols
#         yield self.new_cols
#         for i, row in enumerate(combined_data):
#             y_data = [self.data.rows[i][col.at] for col in self.data.cols.y]
#             new_row = list(row) + y_data
#             yield new_row
            
    
    
# class FAMDProcessor:
#     def __init__(self, data: DATA, n_components: int):
#         self.data = data
#         self.n_components = n_components
#         self.famd = FAMD(n_components=n_components)
#         self.num_cols = [col.txt for col in self.data.cols.x if isinstance(col, NUM)]
#         self.cat_cols = [col.txt for col in self.data.cols.x if isinstance(col, SYM)]
#         if len(self.num_cols) == 0:
#             self.famd = MCA(n_components=n_components)
#         elif len(self.cat_cols) == 0:
#             self.famd = PCA(n_components=n_components)
#         self.new_cols = []

#     def fit_transform(self) -> DATA:
#         # Extract and preprocess numeric and categorical data
#         combined_data = self._extract_data()
#         # Apply FAMD
#         transformed_data = self.famd.fit_transform(combined_data)
        
#         return DATA().adds(self._create_new_data(transformed_data))

#     def _extract_data(self) -> np.ndarray:
#         # Extract data and return as a DataFrame
#         data_dict = {}
#         for col in self.data.cols.x:
#             col_data = [row[col.at] for row in self.data.rows]
#             if isinstance(col, NUM):
#                 data_dict[col.txt] = col_data
#             else:
#                 data_dict[col.txt] = [str(x) for x in col_data]
#         return pd.DataFrame(data_dict)
    
#     def _combine_data(self, numeric_data: np.ndarray, cat_data: np.ndarray) -> np.ndarray:
#         if numeric_data.size == 0:
#             return cat_data
#         if cat_data.size == 0:
#             return numeric_data
#         return np.hstack((numeric_data, cat_data))


#     def _create_new_data(self, transformed_data):
#         self.new_cols = ['Famd'+str(i) for i in range(self.n_components)] + [col.txt for col in self.data.cols.y]
#         yield self.new_cols
#         if not isinstance(transformed_data, pd.DataFrame):
#             transformed_data = pd.DataFrame(transformed_data)
#         for i, row in transformed_data.iterrows():
#             y_data = [self.data.rows[i][col.at] for col in self.data.cols.y]
#             new_row = list(row) + y_data
#             yield new_row
            

# # train = "/workspaces/ezr/data/optimize/config/SS-W.csv"
# # data = DATA().adds(csv(train))
# # print(len(data.cols.x))
# # pca_processor = PCAProcessor(data, n_components=1, cat_method='one_hot')
# # famd_processor = FAMDProcessor(data, n_components=2)
# # new_data = famd_processor.fit_transform()
# # print("X___________")
# # print(new_data.cols.x)
# # print("Y___________")
# # print(new_data.cols.y)
# # print(len(new_data.cols.x))
# # print(len(new_data.cols.y))