import sys
import os
import logging
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Tuple
from car_price.exception import CarException
from car_price.configuration.mongo_operations import MongoDBOperation
from car_price.entity.config_entity import DataIngestionConfig
from car_price.entity.artifacts_entity import DataIngestionArtifacts
from car_price.constant import TEST_SIZE

logger = logging.getLogger(__name__)



class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, mongo_op: MongoDBOperation):
        self.data_ingestion_config = data_ingestion_config
        self.mongo_op = mongo_op

    def get_data_from_mongodb(self) -> DataFrame:
        logger.info("Entered get_data_from_mongodb method of Data_Ingestion class")
        try:
            logger.info("Getting the dataframe from mongodb")
            df = self.mongo_op.get_collection_as_dataframe(
                self.data_ingestion_config.DB_NAME, self.data_ingestion_config.COLLECTION_NAME
            )
            logger.info("Got the dataframe from mongodb")
            logger.info(
                "Exited the get_data_from_mongodb method of Data_Ingestion class"
            )
            return df

        except Exception as e:
            raise CarException(e, sys) from e


    def split_data_as_train_test(self, df:DataFrame) -> Tuple[DataFrame, DataFrame]:
        logger.info(
            "Entered split_data_as_train_test method of Data_Ingestion class"
        )
        try:
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFCATS_DIR, exist_ok=True)
            train_set, test_set = train_test_split(df, test_size=TEST_SIZE)
            logger.info("Performed train test split on the dataframe")
            os.makedirs(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE_DIR, exist_ok=True)
            logger.info(f"Created {os.path.basename(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE_DIR)} directory.")
            os.makedirs(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR, exist_ok=True)
            logger.info(f"Created {os.path.basename(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR)} directory.")
            train_set.to_csv(self.data_ingestion_config.TRAIN_DATA_FILE_PATH, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.TEST_DATA_FILE_PATH, index=False, header=True)
            logger.info("Converted Train Dataframe and Test Dataframe into csv")            
            logger.info(f"Saved {os.path.basename(self.data_ingestion_config.TRAIN_DATA_FILE_PATH)},\
                 {os.path.basename(self.data_ingestion_config.TEST_DATA_FILE_PATH)} in\
                     {os.path.basename(self.data_ingestion_config.DATA_INGESTION_ARTIFCATS_DIR)}."
            )
            logger.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            return train_set, test_set

        except Exception as e:
            raise CarException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info(
            "Entered initiate_data_ingestion method of Data_Ingestion class"
        )
        try:
            df = self.get_data_from_mongodb()
            df1 = df.drop(self.data_ingestion_config.DROP_COLS, axis=1)
            logger.info("Got the data from mongodb")
            self.split_data_as_train_test(df1)
            logger.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifacts = DataIngestionArtifacts(train_data_file_path=self.data_ingestion_config.TRAIN_DATA_FILE_PATH,
                                                               test_data_file_path=self.data_ingestion_config.TEST_DATA_FILE_PATH)

            return data_ingestion_artifacts
            
        except Exception as e:
            raise CarException(e, sys) from e
