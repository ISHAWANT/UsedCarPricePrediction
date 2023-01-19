import sys
from car_price.configuration.mongo_operations import MongoDBOperation
from car_price.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts, DataValidationArtifacts, ModelEvaluationArtifact, ModelPusherArtifacts, ModelTrainerArtifacts
from car_price.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelEvaluationConfig, ModelPusherConfig, ModelTrainerConfig
from car_price.components.data_ingestion import DataIngestion
from car_price.components.data_transformation import DataTransformation
from car_price.components.data_validation import DataValidation
from car_price.components.model_pusher import ModelPusher
from car_price.components.model_trainer import ModelTrainer
from car_price.components.model_evaluation import ModelEvaluation
from car_price.configuration.s3_operations import S3Operation
from car_price.exception import CarException
import logging

logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.mongo_op = MongoDBOperation()
        self.s3_operations = S3Operation()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logger.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config, mongo_op=self.mongo_op)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Got the train_set and test_set from mongodb")
            logger.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact

        except Exception as e:
            raise CarException(e, sys) from e

    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifacts) -> DataValidationArtifacts:
        logger.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(data_ingestion_artifacts=data_ingestion_artifact, data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info("Performed the data validation operation")
            logger.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )
            return data_validation_artifact

        except Exception as e:
            raise CarException(e, sys) from e


    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifacts) -> DataTransformationArtifacts:
        logger.info(
            "Entered the start_data_transformation method of TrainPipeline class"
        )
        try:
            data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifact, 
                                                        data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )
            return data_transformation_artifact

        except Exception as e:
            raise CarException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config
                                         )
            model_trainer_artifact = model_trainer.initiate_model_trainer()   
            return model_trainer_artifact

        except Exception as e:
            raise CarException(e, sys) from e

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifacts,
                               model_trainer_artifact: ModelTrainerArtifacts) -> ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(model_evaluation_config=self.model_evaluation_config,
                                               data_ingestion_artifact=data_ingestion_artifact,
                                               model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact

        except Exception as e:
            raise CarException(e, sys) from e

    def start_model_pusher(self, model_trainer_artifacts: ModelTrainerArtifacts, s3: S3Operation, 
                            data_transformation_artifacts: DataTransformationArtifacts) -> ModelPusherArtifacts:
        logger.info("Entered the start_model_pusher method of TrainPipeline class")
        try:
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config, model_trainer_artifacts=model_trainer_artifacts, 
                                        s3=s3, data_transformation_artifacts=data_transformation_artifacts)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info("Initiated the model pusher")
            logger.info("Exited the start_model_pusher method of TrainPipeline class")
            return model_pusher_artifact

        except Exception as e:
            raise CarException(e, sys) from e


    def run_pipeline(self) -> None:
        logger.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
            
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                logger.info("Model not accepted")
                return None
            model_pusher_artifact = self.start_model_pusher(model_trainer_artifacts=model_trainer_artifact, 
                                                            s3=self.s3_operations, 
                                                            data_transformation_artifacts=data_transformation_artifact)
            logger.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise CarException(e, sys) from e