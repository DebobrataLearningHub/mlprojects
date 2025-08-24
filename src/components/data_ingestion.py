import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.config import Config
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        self.config = Config()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion START")
            os.makedirs(self.config.base_path, exist_ok=True)
            logging.info("Read data from file")
            df = pd.csv(self.config.input_file_path)
            num_columns = df.select_dtypes(exclude=["object"]).columns.tolist()
            cat_columns = df.select_dtypes(include=["object"]).columns.tolist()
            target_column = df['math_score']
            df.to_csv(self.config.raw_data_path,index=False,header=True)
            train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)
            train_df.to_csv(self.config.raw_data_path,index=False,header=True)
            test_df.to_csv(self.config.raw_data_path,index=False,header=True)
            logging.info("Data Ingestion END")
            return(
                self.config.train_data_path,
                self.config.test_data_path,
                num_columns,
                cat_columns,
                target_column
            )

        except Exception as e:
            logging.error(e)

