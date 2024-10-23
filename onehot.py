from sklearn.preprocessing import OneHotEncoder
import numpy as np
import stats
import sys, random
import ezr
from ezr import *
import pandas as pd


class OneHotProcessor:
    def __init__(self, data: DATA):
        self.data = data
        self.num_cols = [col.txt for col in self.data.cols.x if isinstance(col, NUM)]
        self.cat_cols = [col.txt for col in self.data.cols.x if isinstance(col, SYM)]
        self.new_cols = []
        self.col_map = {name:name for name in self.num_cols}
        self.encoder = OneHotEncoder(sparse_output=False)
    
    def fit_transform(self) -> DATA:
        # Extract symbolic data
        df = self._extract_data()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()        
        # Perform one-hot encoding
        one_hot_encoded = self.encoder.fit_transform(df[categorical_columns])

        one_hot_df = pd.DataFrame(one_hot_encoded, columns=self.encoder.get_feature_names_out(categorical_columns))

        # Concatenate the one-hot encoded dataframe with the original dataframe
        df_encoded = pd.concat([df, one_hot_df], axis=1)

        # Drop the original categorical columns
        df_encoded = df_encoded.drop(categorical_columns, axis=1)
        
        return DATA().adds(self._create_new_data(df_encoded))
    
    def _extract_categorical_data(self) -> pd.DataFrame:
        return self.data.df[self.cat_cols]
    
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
    
    def _create_new_data(self, transformed_data):
        self.new_cols = [col for col in transformed_data.columns]
        self.new_cols = [col.capitalize() for col in self.new_cols]+[col.txt for col in self.data.cols.y]
        yield self.new_cols
        
        for i, row in transformed_data.iterrows():
            y_data = [self.data.rows[i][col.at] for col in self.data.cols.y]
            row = [int(x) for x in row]
            new_row = list(row) + list(y_data)
            yield new_row
    

# Example usage
train = "/workspaces/ezr/data/optimize/config/SS-W.csv"
data = DATA().adds(csv(train))
print(len(data.cols.x))
encoder = OneHotProcessor(data)
new_data = encoder.fit_transform()
print("X___________")
print(new_data.cols.x)
print("Y___________")
print(new_data.cols.y)
print(len(new_data.cols.x))
print(len(new_data.cols.y))