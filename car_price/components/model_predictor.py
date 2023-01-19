import logging
import sys
from typing import Dict
from pandas import DataFrame
import pandas as pd
from car_price.constant import *
from car_price.configuration.s3_operations import S3Operation
from car_price.exception import CarException

logger = logging.getLogger(__name__)

class CarData:
    def __init__(self, car_name, vehicle_age, km_driven, seller_type, fuel_type, transmission_type, mileage, engine, max_power, seats):
        self.car_name = car_name
        self.vehicle_age = vehicle_age
        self.km_driven = km_driven
        self.seller_type = seller_type
        self.fuel_type = fuel_type
        self.transmission_type = transmission_type
        self.mileage = mileage
        self.engine = engine
        self.max_power = max_power
        self.seats = seats


    def get_data(self) -> Dict:
        logger.info("Entered get_data method of SensorData class")
        try:
            input_data = {
                "car_name": [self.car_name],
                "vehicle_age": [self.vehicle_age],
                "km_driven": [self.km_driven],
                "seller_type": [self.seller_type],
                "fuel_type": [self.fuel_type],
                "transmission_type": [self.transmission_type],
                "mileage": [self.mileage],
                "engine": [self.engine],
                "max_power": [self.max_power],
                "seats": [self.seats]
                }
            return input_data
        
        except Exception as e:
            raise CarException(e, sys)


    def get_carprice_input_data_frame(self) -> DataFrame:
        logger.info(
            "Entered get_carprice_input_data_frame method of CarPriceData class"
        )
        try:
            carprice_input_dict = self.get_data()
            logger.info("Got car data as dict")
            logger.info(
                "Exited get_carprice_input_data_frame method of CarPriceData class"
            )
            return pd.DataFrame(carprice_input_dict)

        except Exception as e:
            raise CarException(e, sys) from e


class CarPricePredictor:
    def __init__(self):
        self.s3 = S3Operation()
        self.bucket_name = BUCKET_NAME


    def predict(self, X) -> None:
        logger.info("Entered predict method of CarPricePredictor class")
        try:
            best_model = self.s3.load_model(MODEL_FILE_NAME, self.bucket_name)
            logger.info("Loaded best model from s3 bucket")
            result = best_model.predict(X)
            logger.info("Exited predict method of CarPricePredictor class")
            return result

        except Exception as e:
            raise CarException(e, sys) from e