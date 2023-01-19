# 🚗 Used Car Price Prediction

## Problem statement
There are many used car buyers and sellers in India. The majority of people today have the dream of owning a car because cars have become a necessity. However, people often purchase used cars as a result of a lack of funds. On the other hand, those in need of cash sell their vehicles. The biggest query, however, is how much should a car cost? With the help of a car's attributes and machine learning algorithms, it is now possible to predict an approximate used car price.

## Solution Proposed
It is a regression issue, and predictions are made using data from the Cardheko website, which tracks sales of used cars in India. Numerous regression methods, such as XGboost and Random forest of decision trees, have been explored and constructed a machine learning model that could be integrated into a client's website to allow customers to predict the prices of used cars.

## Dataset used
CarDekho is a review-sharing website that offers images, videos, and reviews of various vehicles that can be purchased in India. For this project, we will be considering the used car section of this Gateway.

## Tech Stack Used
1. Python 
2. FastAPI 
3. Machine learning algorithms
4. Docker
5. MongoDB

## Infrastructure required
1. AWS S3
2. AWS EC2
3. Github Actions

## How to run
Before you run this project make sure you have MongoDB Atlas account and you have the Car dekho dataset into it.

Step 1. Cloning the repository.
```
git clone https://github.com/Machine-Learning-01/car-price-prediction.git
```
Step 2. Create a conda environment.
```
conda create --prefix venv python=3.9 -y
```
```
conda activate venv/
````
Step 3. Install the requirements 
```
pip install -r requirements.txt
```
Step 4. Export the environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

export MONGODB_URL="mongodb+srv://<username>:<password>@ineuron-ai-projects.7eh1w4s.mongodb.net/?retryWrites=true&w=majority"

```
Step 5. Run the application server
```
python app.py
```
Step 6. Train application
```bash
http://localhost:8080/train
```
Step 7. Prediction application
```bash
http://localhost:8080/predict
```
## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image
```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> --build-arg MONGODB_URL=<MONGODB_URL> . 
```

3. Run the Docker image
```
docker run -d -p 8080:8080 <IMAGE_NAME>
```
## Project Architecture - 

![WhatsApp Image 2022-09-22 at 15 29 04](https://user-images.githubusercontent.com/71321529/192722300-b906b222-63f7-452b-8e30-e234405031f2.jpeg)


![WhatsApp Image 2022-09-22 at 15 29 10](https://user-images.githubusercontent.com/71321529/192721926-de265f9b-f301-4943-ac7d-948bff7be9a0.jpeg)

![WhatsApp Image 2022-09-22 at 15 29 19](https://user-images.githubusercontent.com/71321529/192722336-54016f79-89ef-4c8c-9d71-a6e91ebab03f.jpeg)


## Models Used
* Linear Regression
* Lasso Regression
* Ridge Regression
* K-Neighbors Regressor
* Decision Tree
* Random Forest Regressor
* XGBRegressor
* CatBoosting Regressor
* AdaBoost Regressor

From these above models after hyperparameter optimization we selected Top two models which were XGBRegressor and Random Forest Regressors and used the following in Pipeline.

* GridSearchCV is used for Hyperparameter Optimization in the pipeline.

## `car_price` is the main package folder which contains 


**Components** : Contains all components of Machine Learning Project
- DataIngestion
- DataValidation
- DataTransformations
- ModelTrainer

**Custom Logger and Exceptions** are used in the Project for better debugging purposes.

## Conclusion
- This Project can be used in real-life by Users or Used car dealers to predict the Estimated Price of the car based on input specifications.