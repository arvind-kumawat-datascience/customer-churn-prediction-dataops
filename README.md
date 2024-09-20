# Customer Churn Prediction System

## Overview
This project implements a Customer Churn Prediction System using Flask and machine learning techniques. The system enables businesses to predict which customers are likely to churn based on their historical data. Users can upload datasets in Excel or CSV format, and the system processes the data, applies machine learning models, and provides insights into customer churn.

## Features

### 1. Flask Backend
- A web framework that handles:
  - Data ingestion and processing.
  - Model inference for predicting customer churn.
  - Serving results to the user interface.
  
### 2. Machine Learning
- Predictive models are used to:
  - Analyze customer data.
  - Provide forecasts of customer churn based on historical trends.
  - Enable businesses to take proactive measures to retain customers.
  
### 3. File Upload
- Users can upload Excel (`.xlsx`) or CSV (`.csv`) files containing customer data.
- The system processes the uploaded file and returns churn predictions.
  
### 4. User Interface
- A simple, intuitive web interface allows users to:
  - Upload files.
  - View the results of churn prediction.
  - Gain insights into which customers are likely to leave.
  
## Technologies

### 1. Flask
- Flask serves as the backend framework, handling requests, and coordinating data flow between the user interface and machine learning models.

### 2. Python
- Python is used for data processing and implementing machine learning algorithms.

### 3. Scikit-Learn
- The machine learning library used for building churn prediction models, including:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Gradient Boosting (e.g., XGBoost)
  
### 4. Pandas
- Pandas is utilized for data manipulation and analysis. It handles:
  - Loading and cleaning data.
  - Performing necessary transformations before feeding the data into the machine learning models.

### 5. Excel/CSV
- Input datasets are accepted in Excel (`.xlsx`) or CSV (`.csv`) formats.
- These datasets contain customer details such as:
  - Customer ID
  - Demographics (age, gender, etc.)
  - Subscription data (tenure, monthly charges, etc.)
  - Behavior patterns (support tickets, payment history)
  - Target variable: `Churn` (Yes/No)

## Workflow

### 1. Data Upload
- Users upload customer data in Excel or CSV format through the web interface.
  
### 2. Data Preprocessing
- The system performs data cleaning, handles missing values, and prepares the data for model training.
  
### 3. Churn Prediction
- The preprocessed data is passed to the machine learning models for churn prediction.
- The system evaluates each customer's likelihood of churning.
  
### 4. Results Display
- Churn predictions are returned to the user through the web interface.
- Users can download the results for further analysis.

## How to Use the System

### Step 1: Set up the Environment
- Install dependencies:
  ```bash
  pip install flask pandas scikit-learn openpyxl
