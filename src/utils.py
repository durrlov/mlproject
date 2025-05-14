import os
import sys
import pickle
import json

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    try:
        report = []
        best_model = None
        best_score = -np.inf

        for model_name, model in models.items():
            gs= GridSearchCV(estimator= model,
                             param_grid= params[model_name],
                             cv= 3,
                             error_score= 'raise')
            
            gs.fit(X_train, y_train)
            best_estimator= gs.best_estimator_

            y_pred_train= best_estimator.predict(X_train)
            y_pred_test= best_estimator.predict(X_test)

            r2_train= r2_score(y_train, y_pred_train)
            r2_test= r2_score(y_test, y_pred_test)

            #report[model_name]= [best_estimator, r2_train, r2_test]
            report.append({
                'Model Name': model_name,
                'Model': best_estimator,
                'r2_train': r2_train,
                'r2_test': r2_test
            })

            if r2_test > best_score:
                best_model= best_estimator
                best_score= r2_test
            
            logging.info(f'{model_name}, r2_test= {r2_test}')
            
            report_df= pd.DataFrame(report).sort_values(by= 'r2_test', ascending= False)

        return best_model, report_df

    
    except Exception as e:
        raise CustomException(e, sys)




def save_json(file_path, obj):
    try:
        with open(file_path, 'w') as file_obj:
            json.dump(obj, file_obj, indent= 4)

    except Exception as e:
        raise CustomException(e, sys)




def load_json(file_path):
    try:
        with open(file_path, 'r') as file_obj:
            return json.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)