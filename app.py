import os
import sys
import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.predict_rough import PredictPipeline
from src.utils import load_json

app= Flask(__name__)

input_schema_path= os.path.join('artifacts', 'input_schema.json')
input_schema= load_json(file_path= input_schema_path)

cat_columns = input_schema["cat_columns"]
num_columns = input_schema["num_columns"]


@app.route('/')
def index():
    return render_template('index_rough.html')

@app.route('/predictdata', methods= ['GET', 'POST'])
def predict_datapoint():
    if request.method== 'GET':
        return render_template('home_rough.html', cat_columns= cat_columns, num_columns= num_columns)
    
    else:
        input_dict= {}

        for col in cat_columns:
            input_dict[col]= [request.form.get(col)]

        for col in num_columns:
            input_dict[col]= [float(request.form.get(col))]

        df= pd.DataFrame(input_dict)
        print(df)

        predict_pipeline= PredictPipeline()
        results= predict_pipeline.predict(df)

        return render_template('home_rough.html',
                               cat_columns= cat_columns,
                               num_columns= num_columns,
                               results= results[0])
    


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug= True)
