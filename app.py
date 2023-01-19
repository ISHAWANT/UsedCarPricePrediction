from fastapi import FastAPI, Request
from typing import Optional
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from car_price.utils.main_utils import MainUtils

from car_price.components.model_predictor import CarPricePredictor, CarData
from car_price.constant import APP_HOST, APP_PORT
from car_price.pipeline.train_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.car_name: Optional[str] = None
        self.vehicle_age: Optional[str] = None
        self.km_driven: Optional[str] = None
        self.seller_type: Optional[str] = None
        self.fuel_type: Optional[str] = None
        self.transmission_type: Optional[str] = None
        self.mileage: Optional[str] = None
        self.engine: Optional[str] = None
        self.max_power: Optional[str] = None
        self.seats: Optional[str] = None
        

    async def get_car_data(self):
        form =  await self.request.form()
        self.car_name = form.get("car_name")
        self.vehicle_age = form.get("vehicle_age")
        self.km_driven = form.get("km_driven")
        self.seller_type = form.get("seller_type")
        self.fuel_type = form.get("fuel_type")
        self.transmission_type = form.get("transmission_type")
        self.mileage = form.get("mileage")
        self.engine = form.get("engine")
        self.max_power = form.get("max_power")
        self.seats = form.get("seats")

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.get("/predict")
async def predictGetRouteClient(request: Request):
    try:
        utils = MainUtils()

        car_list = utils.get_car_list()

        return templates.TemplateResponse(
            "car_price.html",{"request": request, "context": "Rendering", "car_list": car_list})

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        utils = MainUtils()
        car_list = utils.get_car_list()
        form = DataForm(request)
        await form.get_car_data()
        
        car_price_data = CarData(car_name= form.car_name, 
                                   vehicle_age= form.vehicle_age, 
                                   km_driven= form.km_driven, 
                                   seller_type= form.seller_type, 
                                   fuel_type= form.fuel_type, 
                                   transmission_type= form.transmission_type, 
                                   mileage= form.mileage,
                                   engine= form.engine,
                                   max_power = form.max_power,
                                   seats = form.seats
                                   )
        
        car_price_df = car_price_data.get_carprice_input_data_frame()
        car_price_predictor = CarPricePredictor()
        car_price_value = round(car_price_predictor.predict(X=car_price_df)[0], 2)

        return templates.TemplateResponse(
            "car_price.html",
            {"request": request, "context": car_price_value, "car_list": car_list}
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)