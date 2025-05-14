import os
import sys
from src.exception import CustomException
from src.utils import load_object

import pandas as pd

class PredictPipeline:
    def __init__(self):
        try:
            model_path= os.path.join('artifacts', 'model.pkl')
            preprocessor_path= os.path.join('artifacts', 'preprocessor.pkl')

            self.model= load_object(file_path= model_path)
            self.preprocessor= load_object(file_path= preprocessor_path)

        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, features: pd.DataFrame):
        try:
            data_scaled= self.preprocessor.transform(features)
            preds= self.model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)