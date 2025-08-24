import os
from dataclasses import dataclass


BASE_PATH = "artifactory"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
RAW_FILE_NAME = "data.csv"
INPUT_FILE_DIR = "data"
INPUT_FILE_NAME = "stud.csv"
MODEL_NAME ="model.pkl"
PREPROCESSOR_NAME ="preprocessor.pkl"


@dataclass
class Config:
    base_path: str = BASE_PATH
    train_data_path: str = os.path.join(BASE_PATH, TRAIN_FILE_NAME)
    test_data_path: str = os.path.join(BASE_PATH, TEST_FILE_NAME)
    raw_data_path: str = os.path.join(BASE_PATH, RAW_FILE_NAME)
    input_file_path: str = os.path.join(INPUT_FILE_DIR, INPUT_FILE_NAME)
    preprocessor_file_path: str = os.path.join(BASE_PATH, PREPROCESSOR_NAME)
    model_file_path: str = os.path.join(BASE_PATH, MODEL_NAME)
