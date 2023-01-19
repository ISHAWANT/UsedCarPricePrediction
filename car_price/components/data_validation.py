import json
import logging
import sys
import os
import pandas as pd
from pandas import DataFrame
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from typing import Tuple, Union
from car_price.exception import CarException
from car_price.entity.config_entity import DataValidationConfig
from car_price.entity.artifacts_entity import DataIngestionArtifacts, DataValidationArtifacts

logger = logging.getLogger(__name__)


class DataValidation:
    def __init__(self, data_ingestion_artifacts: DataIngestionArtifacts, data_validation_config: DataValidationConfig):
        self.data_ingestion_atifacts = data_ingestion_artifacts
        self.data_validation_config = data_validation_config


    def validate_schema_columns(self, df: DataFrame) -> bool:
        try:
            if len(df.columns) == len(self.data_validation_config.SCHEMA_CONFIG['columns']):                       
                validation_status = True
            else:
                validation_status = False
            return validation_status

        except Exception as e:
            raise CarException(e, sys) from e


    def is_numerical_column_exists(self, df: DataFrame) -> bool:
        try:
            validation_status = False
            for column in self.data_validation_config.SCHEMA_CONFIG["numerical_columns"]:
                if column not in df.columns:
                    logger.info(f"Numerical column - {column} not found in dataframe")
                else:
                    validation_status = True
            return validation_status

        except Exception as e:
            raise CarException(e, sys) from e        


    def is_categorical_column_exists(self, df: DataFrame) -> bool:
        try:
            validation_status = False
            for column in self.data_validation_config.SCHEMA_CONFIG["categorical_columns"]:
                if column not in df.columns:
                    logger.info(f"categorical column - {column} not found in dataframe")
                else:
                    validation_status = True
            return validation_status

        except Exception as e:
            raise CarException(e, sys) from e   


    def validate_dataset_schema_columns(self) -> Tuple[bool, bool]:
        logger.info(
            "Entered validate_dataset_schema_columns method of Data_Validation class"
        )
        try:
            logger.info("Validating dataset schema columns")
            train_schema_status = self.validate_schema_columns(self.train_set)
            logger.info("Validated dataset schema columns on the train set")
            test_schema_status = self.validate_schema_columns(self.test_set)
            logger.info("Validated dataset schema columns on the test set")
            logger.info("Validated dataset schema columns")
            return train_schema_status, test_schema_status

        except Exception as e:
            raise CarException(e, sys) from e


    def validate_is_numerical_column_exists(self) -> Tuple[bool, bool]:
        logger.info(
            "Entered validate_dataset_schema_for_numerical_datatype method of Data_Validation class"
        )
        try:
            logger.info("Validating dataset schema for numerical datatype")
            train_num_datatype_status = self.is_numerical_column_exists(
                self.train_set
            )
            logger.info(
                "Validated dataset schema for numerical datatype for train set"
            )
            test_num_datatype_status = self.is_numerical_column_exists(
                self.test_set
            )
            logger.info("Validated dataset schema for numerical datatype for test set")
            logger.info(
                "Exited validate_dataset_schema_for_numerical_datatype method of Data_Validation class"
            )
            return train_num_datatype_status, test_num_datatype_status

        except Exception as e:
            raise CarException(e, sys) from e


    def validate_is_categorical_column_exists(self) -> Tuple[bool, bool]:
        logger.info(
            "Entered validate_dataset_schema_for_numerical_datatype method of Data_Validation class"
        )
        try:



            logger.info("Validating dataset schema for numerical datatype")
            train_cat_datatype_status = self.is_categorical_column_exists(
                self.train_set
            )
            logger.info(
                "Validated dataset schema for numerical datatype for train set"
            )
            test_cat_datatype_status = self.is_categorical_column_exists(
                self.test_set
            )
            logger.info("Validated dataset schema for numerical datatype for test set")
            logger.info(
                "Exited validate_dataset_schema_for_numerical_datatype method of Data_Validation class"
            )
            return train_cat_datatype_status, test_cat_datatype_status

        except Exception as e:
            raise CarException(e, sys) from e


    def detect_dataset_drift(self, reference: DataFrame, production: DataFrame, get_ratio: bool = False) -> Union[bool, float]:
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference, production)
            report = data_drift_profile.json()
            json_report = json.loads(report)
            data_drift_file_path = self.data_validation_config.DATA_DRIFT_FILE_PATH
            self.data_validation_config.UTILS.write_json_to_yaml_file(json_report, data_drift_file_path)
            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]
            if get_ratio:
                return n_drifted_features / n_features
            else:
                return json_report["data_drift"]["data"]["metrics"]["dataset_drift"]

        except Exception as e:
            raise CarException(e, sys) from e


    def initiate_data_validation(self) -> DataValidationArtifacts:
        logger.info("Entered initiate_data_validation method of Data_Validation class")
        try:
            self.train_set = pd.read_csv(self.data_ingestion_atifacts.train_data_file_path)
            self.test_set = pd.read_csv(self.data_ingestion_atifacts.test_data_file_path)
            logger.info("Initiated data validation for the dataset")
            os.makedirs(self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR, exist_ok=True)
            logger.info(f"Created Artifatcs directory for {os.path.basename(self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR)}")
            drift = self.detect_dataset_drift(self.train_set, self.test_set)

            (
                schema_train_col_status,
                schema_test_col_status,
            ) = self.validate_dataset_schema_columns()
            logger.info(
                f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}"
            )
            logger.info("Validated dataset schema columns")
            (
                schema_train_cat_cols_status,
                schema_test_cat_cols_status,
            ) = self.validate_is_categorical_column_exists()
            logger.info(
                f"Schema train cat cols status is {schema_train_cat_cols_status} and schema test cat cols status is {schema_test_cat_cols_status}"
            )
            logger.info("Validated dataset schema for catergorical datatype")
            (
                schema_train_num_cols_status,
                schema_test_num_cols_status,
            ) = self.validate_is_numerical_column_exists()
            logger.info(
                f"Schema train numerical cols status is {schema_train_num_cols_status} and schema test numerical cols status is {schema_test_num_cols_status}"
            )
            logger.info("Validated dataset schema for numerical datatype")

            drift_status = None
            if (
                schema_train_cat_cols_status is True
                and schema_test_cat_cols_status is True
                and schema_train_num_cols_status is True
                and schema_test_num_cols_status is True
                and schema_train_col_status is True
                and schema_test_col_status is True
                and drift is False
            ):
                logger.info("Dataset schema validation completed")
                drift_status == True
            else:
                drift_status == False

            data_validation_artifacts = DataValidationArtifacts(data_drift_file_path=self.data_validation_config.DATA_DRIFT_FILE_PATH, 
                                                                    validation_status=drift_status)

            return data_validation_artifacts

        except Exception as e:
            raise CarException(e, sys) from e
