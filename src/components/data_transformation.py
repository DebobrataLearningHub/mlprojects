import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.config import Config
from src.utility import create_data_formation_object
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataTransormation:
    def __init__(self):
        self.config = Config()    

    def initiate_data_transformation(self, num_columns, cat_columns, target_column):
        try:
            logging.info("Data Transformation START")
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)
            preprocessor = create_data_formation_object(
                num_columns, cat_columns)
            train_input_df = train_df.drop(columns=[target_column], axis=1)
            train_target_df = train_df[target_column]
            test_input_df = test_df.drop(columns=[target_column], axis=1)
            test_target_df = test_df[target_column]

            train_input_df_arr = preprocessor.fit_transform(train_input_df)
            test_input_df_arr = preprocessor.transform(test_input_df)

            train_arr = np.c_[train_input_df_arr, np.array(train_target_df)]
            test_arr = np.c_[test_input_df_arr, np.array(test_target_df)]
            logging.info("Data Transformation END")
            return(
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.error(e)
