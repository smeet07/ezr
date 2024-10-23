from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import stats
import sys, random
import ezr
from ezr import the, DATA, NUM, SYM, COL, csv, dot, adds
from prince import FAMD, MCA
import pandas as pd


class PCAProcessor:
    def __init__(self, data: DATA, n_components: int, cat_method: str = 'none'):
        self.data = data
        self.n_components = n_components
        self.cat_method = cat_method
        self.pca = PCA(n_components=n_components)
        self.encoder = None
        self.num_cols = [col.txt for col in self.data.cols.x if isinstance(col, NUM)]
        self.cat_cols = [col.txt for col in self.data.cols.x if isinstance(col, SYM)]
        self.new_cols = []

    def fit_transform(self) -> DATA:
        # Extract and preprocess numeric and categorical data
        numeric_data = self._extract_numeric_data()
        cat_data = self._extract_categorical_data()
        print("number of numeric columns: ", len(numeric_data[0]))
        print("number of categorical columns: ", len(cat_data[0]))
        if self.cat_method == 'one_hot':
            cat_data = self._one_hot_encode(cat_data)
        elif self.cat_method == 'label':
            cat_data = self._label_encode(cat_data)
        
        # Combine numeric and categorical data
        combined_data = self._combine_data(numeric_data, cat_data)
    
        if self.cat_method == 'none':
            n_samples, n_features = numeric_data.shape
            print("number of numeric columns: ", n_features)
            pca_result = cat_data
            if n_features>0 and n_samples > 0:
                pca_result= self.pca.fit_transform(numeric_data)
                pca_result = self._combine_data(pca_result, cat_data)
        else:
            n_samples, n_features = combined_data.shape
            print("number of combined columns: ", n_features, "after ", self.cat_method)
            if n_features>0 and n_samples > 0:
                pca_result= self.pca.fit_transform(combined_data)

        pca_result_samples, pca_result_features = pca_result.shape   
        print("number of pca columns: ", pca_result_features)    
        # Create new DATA object with the PCA result
        return DATA().adds(self._create_new_data_object(pca_result))

    def _extract_numeric_data(self) -> np.ndarray:
        return np.array([[row[col.at] for col in self.data.cols.x if isinstance(col, NUM)] for row in self.data.rows])

    def _extract_categorical_data(self) -> np.ndarray:
        return np.array([[row[col.at] for col in self.data.cols.x if isinstance(col, SYM)] for row in self.data.rows])

    def _one_hot_encode(self, cat_data) -> np.ndarray:
        self.encoder = OneHotEncoder(sparse_output=False)
        encoded=self.encoder.fit_transform(cat_data)
        return encoded

    def _label_encode(self,cat_data) -> np.ndarray:
        self.encoder = LabelEncoder()
        return np.array([self.encoder.fit_transform(col) for col in cat_data.T]).T

    def _combine_data(self, numeric_data: np.ndarray, cat_data: np.ndarray) -> np.ndarray:
        if numeric_data.size == 0:
            return cat_data
        if cat_data.size == 0:
            return numeric_data
        return np.hstack((numeric_data, cat_data))

    def _create_new_data_object(self, combined_data: np.ndarray) -> DATA:
        y_cols = [col.txt for col in self.data.cols.y]
        if self.cat_method == 'none':
            self.new_cols = ['Pca'+str(i) for i in range(min(self.n_components, len(self.num_cols)))]
            self.new_cols += self.cat_cols
        else:
            self.new_cols = ['Pca'+str(i) for i in range(self.n_components)]
        

        self.new_cols += y_cols
        yield self.new_cols
        for i,row in enumerate(combined_data):
            y_data = [self.data.rows[i][col.at] for col in self.data.cols.y]
            new_row = list(row) + list(y_data)
            yield new_row
            
    
    
class FAMDProcessor:
    def __init__(self, data: DATA, n_components: int):
        self.data = data
        self.n_components = n_components
        self.famd = FAMD(n_components=n_components)
        self.num_cols = [col.txt for col in self.data.cols.x if isinstance(col, NUM)]
        self.cat_cols = [col.txt for col in self.data.cols.x if isinstance(col, SYM)]
        if len(self.num_cols) == 0:
            self.famd = MCA(n_components=n_components)
        elif len(self.cat_cols) == 0:
            self.famd = PCA(n_components=n_components)
        self.new_cols = []

    def fit_transform(self) -> DATA:
        # Extract and preprocess numeric and categorical data
        combined_data = self._extract_data()
        # Apply FAMD
        transformed_data = self.famd.fit_transform(combined_data)
        
        return DATA().adds(self._create_new_data(transformed_data))

    def _extract_data(self) -> np.ndarray:
        # Extract data and return as a DataFrame
        data_dict = {}
        for col in self.data.cols.x:
            col_data = [row[col.at] for row in self.data.rows]
            if isinstance(col, NUM):
                data_dict[col.txt] = col_data
            else:
                data_dict[col.txt] = [str(x) for x in col_data]
        return pd.DataFrame(data_dict)
    
    def _combine_data(self, numeric_data: np.ndarray, cat_data: np.ndarray) -> np.ndarray:
        if numeric_data.size == 0:
            return cat_data
        if cat_data.size == 0:
            return numeric_data
        return np.hstack((numeric_data, cat_data))


    def _create_new_data(self, transformed_data):
        self.new_cols = ['Famd'+str(i) for i in range(self.n_components)] + [col.txt for col in self.data.cols.y]
        yield self.new_cols
        if not isinstance(transformed_data, pd.DataFrame):
            transformed_data = pd.DataFrame(transformed_data)

        for i, row in transformed_data.iterrows():
            y_data = [self.data.rows[i][col.at] for col in self.data.cols.y]
            new_row = list(row) + list(y_data)
            yield new_row
            

train = "/workspaces/ezr/data/optimize/config/SS-W.csv"
data = DATA().adds(csv(train))
print(len(data.cols.x))
pca_processor = PCAProcessor(data, n_components=1, cat_method='one_hot')
famd_processor = FAMDProcessor(data, n_components=2)
new_data = famd_processor.fit_transform()
print("X___________")
print(new_data.cols.x)
print("Y___________")
print(new_data.cols.y)
print(len(new_data.cols.x))
print(len(new_data.cols.y))