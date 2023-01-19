from dataclasses import dataclass
import logging
import sys
import pandas as pd
from car_price.exception import CarException
from car_price.constant import *
from car_price.entity.artifacts_entity import DataIngestionArtifacts, ModelEvaluationArtifact, ModelTrainerArtifacts
from car_price.entity.config_entity import ModelEvaluationConfig

logger = logging.getLogger(__name__)

@dataclass
class EvaluateModelResponse:
    trained_model_r2_score: float
    s3_model_r2_score: float
    is_model_accepted: bool
    difference: float    


class ModelEvaluation:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifacts,
                        model_evaluation_config: ModelEvaluationConfig, 
                        data_ingestion_artifact: DataIngestionArtifacts):

        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artifact = data_ingestion_artifact


    def get_s3_model(self) -> object:
        try:
            status = self.model_evaluation_config.S3_OPERATIONS.is_model_present(BUCKET_NAME, S3_MODEL_NAME)

            if status == True:
                model = self.model_evaluation_config.S3_OPERATIONS.load_model(MODEL_FILE_NAME, BUCKET_NAME)
                return model

            else:
                return None

        except Exception as e:
            raise CarException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            trained_model = self.model_evaluation_config.UTILS.load_object(self.model_trainer_artifact.trained_model_file_path)
            y_hat_trained_model = trained_model.predict(x)
            trained_model_r2_score = self.model_evaluation_config.UTILS.get_model_score(y, y_hat_trained_model)

            s3_model_r2_score = None
            s3_model = self.get_s3_model()
            if s3_model is not None:
                y_hat_s3_model = s3_model.predict(x)
                s3_model_r2_score = self.model_evaluation_config.UTILS.get_model_score(y, y_hat_s3_model)

            tmp_best_model_score = 0 if s3_model_r2_score is None else s3_model_r2_score

            result = EvaluateModelResponse(trained_model_r2_score=trained_model_r2_score, 
                                            s3_model_r2_score=s3_model_r2_score, 
                                            is_model_accepted=trained_model_r2_score > tmp_best_model_score,
                                            difference=trained_model_r2_score - tmp_best_model_score)

            return result

        except Exception as e:
            raise CarException(e, sys) from e
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_reaponse = self.evaluate_model()

            model_evaluataion_artifact = ModelEvaluationArtifact(is_model_accepted=evaluate_model_reaponse.is_model_accepted,  
                                                                trained_model_path= self.model_trainer_artifact.trained_model_file_path,
                                                                changed_accuracy=evaluate_model_reaponse.difference)
            return model_evaluataion_artifact

        except Exception as e:
            raise CarException(e, sys) from e

        