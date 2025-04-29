import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object, evaluate_models

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models= {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'K Neighbor Regressor': KNeighborsRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Cat Boost Regressor': CatBoostRegressor(),
                'XGB Regressor': XGBRegressor()
            }

            params= {
                'Linear Regression': {},
                'Ridge': {
                    'alpha': np.array(np.logspace(-3, 1, 5, endpoint= True))
                    #'alpha': [1.0]
                },
                'Lasso': {
                    'alpha': np.array(np.logspace(-3, 1, 5, endpoint= True))
                    #'alpha': [1.0]
                },
                'Decision Tree Regressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [2, 3, 5, 10]
                },
                'K Neighbor Regressor': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                'AdaBoost Regressor': {
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.01, 0.05, 0.1, 1]
                },
                'Gradient Boosting Regressor':{
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.01, 0.05, 0.1, 1],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                'Random Forest Regressor':{
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Cat Boost Regressor': {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 1],
                    'depth': [3, 5, 7]
                },
                'XGB Regressor':{
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.01, 0.05, 0.1, 1]
                }   
            }

            best_model, report_df = evaluate_models(
                X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test,
                models= models, params= params
            )

            logging.info(f'Best Model: {best_model}')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            logging.info(f"Transforming the evaluation reports into a DataFrame")
            

            return best_model, report_df


        except Exception as e:
            raise CustomException(e, sys)