# End to End Machine Learning Project
**Developed by**: [S M Bakhtiar](https://www.linkedin.com/in/durrlov/)  
bakhtiar.scr@gmail.com

## Important Project Links
### [Try Real-Time Predictions 🔗](https://render-mlproject.onrender.com/)

[Exploratory Data Analysis Notebook 🔗](https://github.com/durrlov/mlproject/blob/main/notebook/EDA.ipynb)

[Model Training Notebook 🔗](https://github.com/durrlov/mlproject/blob/main/notebook/Model%20Training.ipynb)

## Table of contents
- [1. Introduction](#introduction)
- [2. Problem Statement](#problem)
- [3. Dataset Overview](#dataset)
- [4. Exploratory Data Analysis (EDA)](#eda)
- [5. Data Preprocessing & Feature Engineering](#preprocessing)
- [6. Model Selection & Training](#model)
- [7. Model Evaluation](#evaluation)
- [8. Conclusion & Insights](#conclusion)
- [9. Deployment](#deployment)
- [10. Modular Project Architecture & Engineering Practices](#modular)
    - [10.1. Components Breakdown](#components)
    - [10.2. Utility and Support Modules](#support)
    - [10.3. Configuration & Packaging](#configuration)
- [11. Summary](#summary)

 

## 1. Introduction <a name= "introduction"></a>
This project showcases an end-to-end machine learning pipeline to predict students' math scores based on demographic and academic performance features. The project follows best practices for model building, evaluation, and deployment.

## 2. Problem Statement<a name= "problem"></a>
The objective is to predict a student’s math score based on demographic features (e.g., gender, race/ethnicity, parental education level, lunch type, test preparation), and performance in reading and writing. This model can help educators identify students at risk and offer targeted support.

## 3. Dataset Overview<a name= "dataset"></a>
- Source:
    - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
- Target Variable: 
    - math_score
- Features:
    - gender (categorical)
    - race_ethnicity (categorical)
    - parental_level_of_education (categorical)
    - lunch (categorical)
    - test_preparation_course (categorical)
    - reading_score (numerical)
    - writing_score (numerical)
- Rows: ~1,000 students
- Missing Values: None

## 4. Exploratory Data Analysis (EDA)<a name= "eda"></a>
- Analyzed the students proportion in different categories of gender, race_ethnicity, parental_level_of_education, lunch, and test_preparation_course
- Analyzed distributions of math_score, reading_score, and writing_score
- Boxplots and barplots used to visualize group-wise performance

## 5. Data Preprocessing & Feature Engineering<a name= "preprocessing"></a>
- Categorical Encoding: OneHotEncoder for all categorical variables
- Feature Scaling: StandardScaler for numeric scores
- Pipeline: ColumnTransformer used to combine transformations
- Artifacts Saved: Preprocessor pipeline saved using pickle for reuse during inference

## 6. Model Selection & Training<a name= "model"></a>
- Models evaluated:
    - LinearRegression
    - Ridge & Lasso
    - DecisionTreeRegressor
    - KNeighborRegressor
    - AdaBoostRegressor
    - GradientBoostingRegressor
    - RandomForestRegressor
    - CatBoostRegressor
    - XGBRegressor
- Hyperparameter Tuning: GridSearchCV
- Train-Test Split: Used train_test_split() with appropriate random seed
- Final model serialized for deployment

## 7. Model Evaluation<a name= "evaluation"></a>
- Metrics Used:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score
- Best model:
    - Lasso (alpha= 0.01) provided high R² (~ 88.06) and low RMSE on test set
- Evaluated model predictions with scatter plots and residuals

## 8. Conclusion & Insights<a name= "conclusion"></a>
- Female students perform better than male students
- Students with standard lunch do better performance than the students with free/reduced lunch
- Finishing preparation course is benefitial
- Students from race/ethnicity of Group E tends to perform well

## 9. Deployment<a name= "deployment"></a>
- Created a Flask app to serve predictions
- CustomData class used to collect and format user input
- PredictPipeline class used to load saved model and preprocessor
- Simple HTML form captures user data
- Hosted on Render with automatic deployment from GitHub

## 10. Modular Project Architecture & Engineering Practices<a name= "modular"></a>
This project maintains industry standards by following a modular and scalable architecture:

#### 10.1. Components Breakdown<a name= "components"></a>
- data_ingestion.py  
  Handles reading raw data from source (e.g., CSV), splitting it into training and testing sets, and saving these artifacts in the artifacts/ directory.

- data_transformation.py  
  Builds preprocessing pipelines using ColumnTransformer, handles categorical and numerical processing, and serializes the preprocessor object for later inference.

- model_trainer.py  
  Encapsulates model training, evaluation, and hyperparameter tuning. Supports multiple regression algorithms and exports the best-performing model.

- input_schema.py  
  Extract feature names and categories and, saves as JSON file in artifacts in the artifacts/ directory for later use to automate the input generation form in the frontend.

- predict_pipeline.py (located in src/pipeline/)  
  Loads the saved model and preprocessor to make predictions on new user input collected via the web form.

#### 10.2. Utility and Support Modules<a name= "support"></a>
- logger.py  
  Custom logging setup to track pipeline progress, errors, and debugging information.

- exception.py  
  A custom exception class to wrap and raise errors cleanly across modules for better traceability.

- utils.py  
  Includes reusable functions like saving and loading objects (pickle), evaluating models, and other helper operations.

#### 10.3. Configuration & Packaging<a name= "configuration"></a>
- setup.py  
  Makes the project installable as a package. Useful for deployment and future integrations.

- requirements.txt  
  Lists all Python dependencies used in the project.

## 11. Summary<a name= "summary"></a>
This project demonstrates a robust, industry-standard machine learning pipeline from data ingestion to deployment. It combines solid EDA, modular engineering practices, and model explainability, making it suitable for both real-world use and demonstrating technical skill in  professional settings.
