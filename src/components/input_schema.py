import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_json

import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class InputSchemaConfig:
    input_schema_file_path:str= os.path.join('artifacts', 'input_schema.json')


class InputSchema:
    def __init__(self):
        self.input_schema_config= InputSchemaConfig()

    def initiate_input_schema(self, train_path):
        try:
            train_df = pd.read_csv(train_path)

            target_column_name = "math_score"
            input_feature = train_df.drop(target_column_name, axis= 1)

            cat_columns = [col for col in input_feature.columns if input_feature[col].dtype == 'O']
            num_columns = [col for col in input_feature.columns if input_feature[col].dtype != 'O']

            input_schema={
                "cat_columns":{
                    col: sorted(input_feature[col].dropna().unique().tolist())
                    for col in cat_columns
                },

                "num_columns": num_columns
            }

            logging.info("Saving input_schema.json")
            save_json(file_path=self.input_schema_config.input_schema_file_path,
                      obj= input_schema)
            
            return self.input_schema_config.input_schema_file_path


        except Exception as e:
            raise CustomException(e, sys)