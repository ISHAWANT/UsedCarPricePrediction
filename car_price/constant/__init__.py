import os
from os import environ
from datetime import datetime

from from_root.root import from_root

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

TARGET_COLUMN = 'selling_price'

CONFIG_FILE_PATH = "config/config.yaml"
SCHEMA_FILE_PATH = "config/schema.yaml"

DB_NAME = 'ineuron'
COLLECTION_NAME = 'car'
DB_URL = environ["MONGODB_URL"]

TEST_SIZE = 0.2

MODEL_CONFIG_FILE = 'config/model.yaml'

ARTIFACTS_DIR = os.path.join(from_root(), 'artifacts', TIMESTAMP)
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'car_price.log'

DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'Train'
DATA_INGESTION_TEST_DIR = 'Test'
DATA_INGESTION_TRAIN_FILE_NAME = 'train.csv'
DATA_INGESTION_TEST_FILE_NAME = 'test.csv'

DATA_VALIDATION_ARTIFACT_DIR = 'DataValidationArtifacts'
DATA_DRIFT_FILE_NAME = "DataDriftReport.yaml"

DATA_TRANSFORMATION_ARTIFCATS_DIR = 'DataTransformationArtifacts'
TRANSFORMED_TRAIN_DATA_DIR = 'TransformedTrain'
TRANSFORMED_TEST_DATA_DIR = 'TransformedTest'
TRANSFORMED_TRAIN_DATA_FILE_NAME = 'transformed_train_data.npz'
TRANSFORMED_TEST_DATA_FILE_NAME = 'transformed_test_data.npz'
PREPROCESSOR_OBJECT_FILE_NAME = "car_price_preprocessor.pkl"

MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
MODEL_FILE_NAME = 'car_price_model.pkl'

BUCKET_NAME = 'car-price-io-files'
S3_MODEL_NAME = 'car_price_model.pkl'



MODEL_SAVE_FORMAT = '.pkl'

APP_HOST = '0.0.0.0'
APP_PORT = 8080