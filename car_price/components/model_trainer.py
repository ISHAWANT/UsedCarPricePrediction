import os
import logging
import sys
import pandas as pd
from typing import List, Tuple
from pandas import DataFrame
from car_price.entity.config_entity import ModelTrainerConfig
from car_price.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts
from car_price.exception import CarException

logger = logging.getLogger(__name__)

class CarPriceModel:
    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X) -> DataFrame:
        logger.info("Entered predict method of CarPriceModel class")
        try:
            logger.info("Using the trained model to get predictions")
            transformed_feature = self.preprocessing_object.transform(X)
            logger.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CarException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config


    def get_trained_models(self, x_data: DataFrame, y_data: DataFrame) -> List[Tuple[float, object, str]]:
        logger.info("Entered get_trained_models method of ModelTrainer class")
        try:
            model_config = self.model_trainer_config.UTILS.read_model_config_file()
            models_list = list(model_config["train_model"].keys())
            logger.info("Got model list from the config file")
            x_train, y_train, x_test, y_test = (
                x_data.drop(x_data.columns[len(x_data.columns)-1], axis=1),
                x_data.iloc[:, -1],
                y_data.drop(y_data.columns[len(y_data.columns)-1], axis=1),
                y_data.iloc[:, -1],
            )
            tuned_model_list = [(self.model_trainer_config.UTILS.get_tuned_model(model_name, x_train, y_train, x_test, y_test)) for model_name in models_list]
            logger.info("Got trained model list")
            logger.info("Exited the get_trained_models method of ModelFinder class")
            return tuned_model_list

        except Exception as e:
            raise CarException(e, sys) from e


    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logger.info(
            "Entered initiate_model_trainer nethod of ModelTrainer class"
        )
        try:
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            logger.info(f"Created artifacts directory for {os.path.basename(self.model_trainer_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")

            train_array = self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            train_df = pd.DataFrame(train_array)
            logger.info(f"Loaded train array from DataTransformationArtifacts directory and converted into Dataframe.")

            test_array = self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            test_df = pd.DataFrame(test_array)
            logger.info(f"Loaded test array from DataTransformationArtifacts directory and converted into Dataframe.")

            list_of_trained_models = self.get_trained_models(train_df, test_df)
            logger.info(
                "Got a list of tuple of model score,model and model name"
            )
            (
                best_model,
                best_model_score,
            ) = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(list_of_trained_models)
            logger.info("Got best model score,model and model name")
            preprocessor_obj_file_path = self.data_transformation_artifact.transformed_object_file_path
            preprocessing_obj = self.model_trainer_config.UTILS.load_object(preprocessor_obj_file_path)
            logger.info("Loaded preprocessing object")
            model_config = self.model_trainer_config.UTILS.read_model_config_file()
            base_model_score = float(model_config['base_model_score'])
            if best_model_score >= base_model_score:
                self.model_trainer_config.UTILS.update_model_score(best_model_score)
                logger.info("Updating model score in yaml file")
                carprice_model = CarPriceModel(preprocessing_obj, best_model)
                logger.info(
                    "Created car price model object with preprocessor and model"
                )
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                logger.info("Created best model file path")
                model_file_path = self.model_trainer_config.UTILS.save_object(trained_model_path, carprice_model)
                logger.info("Saved the best model object path")
            else:
                logger.info("No best model found with score more than base score")
                raise "No best model found with score more than base score "

            model_trainer_artifacts = ModelTrainerArtifacts(trained_model_file_path=model_file_path)

            return model_trainer_artifacts

        except Exception as e:
            raise CarException(e, sys) from e


