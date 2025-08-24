import os
import sys
import pandas as pd
from src.config import Config
from src.logger import logging
from src.exception import CustomException
from src.utility import load_object


class PredictionPipeline:
    def __init__(self):
        self.config = Config()

    def initiate_prediction_pipeline(self, data):
        try:
            model = load_object(self.config.model_file_path)
            preprocessor = load_object(self.config.preprocessor_file_path)
            scaler_data = preprocessor.transform(data)
            predic = model.predict(scaler_data)
            return predic
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def create_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
