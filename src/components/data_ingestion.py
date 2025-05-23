import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.input_schema import InputSchema

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Read the dataset as a dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok= True) 
            # os.path.dirname to get the parent folder (artifacts) of the self.data_ingestion_config.train_data_path (train.csv/test.csv/data.csv)
            
            df.to_csv(self.data_ingestion_config.raw_data_path, index= False, header= True)
            logging.info('Raw data has been saved to artifacts/data.csv')

            logging.info('Train Test Split has been initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state= 42)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index= False, header= True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index= False, header= True)
            logging.info('Ingestion of the data is completed')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        

if __name__== '__main__':
    obj= DataIngestion()
    train_path, test_path= obj.initiate_data_ingestion()

    data_transformation= DataTransformation()
    train_arr, test_arr, _= data_transformation.initiate_data_transformation(train_path, test_path)

    input_schema= InputSchema()
    input_schema_path= input_schema.initiate_input_schema(train_path)

    model_trainer= ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array= train_arr, test_array= test_arr), sep= '\n')
    