# Customer Churn Prediction using Machine Learning and Deep Learning

An end-to-end customer churn prediction system built using Machine Learning and Deep Learning. The project uses the Telco Customer Churn dataset, compares Logistic Regression with a TensorFlow/Keras Deep Neural Network, and provides predictions through a FastAPI backend with a Streamlit frontend.

## Problem Statement

Customer churn is costly for subscription-based businesses. This project predicts whether a customer is likely to churn based on demographic information, subscribed services, contract details, and billing information.

The target variable is `Churn`, which is converted from `Yes`/`No` to `1`/`0`.

## Objectives

* Analyze and preprocess the Telco Customer Churn dataset.
* Train and evaluate a Logistic Regression classification model.
* Train a Deep Neural Network using TensorFlow/Keras.
* Compare both models using accuracy, precision, recall, F1-score, and ROC-AUC.
* Save the trained models and preprocessing pipeline.
* Build a FastAPI REST API for model predictions.
* Build a Streamlit frontend that communicates with the FastAPI backend.
* Provide churn probability and risk classification for new customers.

## Dataset

The project uses the Telco Customer Churn dataset containing 7,043 customer records and 21 columns.

The dataset includes information such as:

* Customer demographics
* Tenure
* Phone and internet services
* Online security and support services
* Contract type
* Payment method
* Monthly charges
* Total charges

`customerID` is excluded from model training because it is only an identifier.

`TotalCharges` is converted to numeric format and missing values are handled during preprocessing.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow/Keras
* FastAPI
* Uvicorn
* Streamlit
* Joblib

## Machine Learning Models

### Logistic Regression

Logistic Regression is used as the baseline classification model because it is simple, interpretable, and effective for binary classification problems.

The model was also tuned using cross-validation over different `C` values with ROC-AUC as the evaluation metric.

### Deep Neural Network

A feedforward Deep Neural Network was developed using TensorFlow/Keras.

The network uses:

* Dense layers
* ReLU activation
* Dropout
* Sigmoid output
* Early stopping

The DNN provides a second modeling approach for comparison with the traditional Machine Learning model.

## Data Preprocessing

The preprocessing pipeline includes:

1. Removing the customer identifier.
2. Converting `TotalCharges` into numeric format.
3. Handling missing values.
4. Separating numerical and categorical features.
5. Standardizing numerical features using `StandardScaler`.
6. Encoding categorical features using `OneHotEncoder`.
7. Fitting the preprocessing pipeline only on the training data to avoid data leakage.

The target variable `Churn` is converted into binary values:

* `0` → No Churn
* `1` → Churn

## Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Data Preprocessing
   ↓
 ┌───────────────────────┐
 │                       │
 ↓                       ↓
Logistic Regression     DNN
 │                       │
 └───────────┬───────────┘
             ↓
       Model Evaluation
             ↓
      Save Trained Models
             ↓
        FastAPI Backend
             ↓
       Streamlit Frontend
             ↓
      Churn Prediction
```

## Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* ROC Curve

The project does not assume that the Deep Neural Network will automatically perform better. The models are compared using evaluation metrics, with ROC-AUC used as an important metric for model comparison.

Training generates evaluation artifacts inside the `reports/` directory.

## Saved Models

The trained components are stored inside the `models/` directory:

```text
models/
├── logistic_regression.pkl
├── churn_dnn.keras
└── preprocessor.pkl
```

The saved preprocessing pipeline ensures that new customer inputs are transformed consistently with the data used during model training.

## FastAPI Backend

FastAPI provides the REST API used for making predictions.

### API Endpoints

#### Health Check

```text
GET /health
```

Checks whether the API is running correctly.

#### Prediction

```text
POST /predict
```

Accepts customer information and returns:

* Selected model
* Churn probability
* Churn risk prediction

Example response:

```json
{
  "model": "logistic_regression",
  "churn_probability": 0.119,
  "prediction": "Low Risk of Churn"
}
```

FastAPI also validates incoming customer information using Pydantic, helping prevent invalid values from reaching the model.

## Streamlit Frontend

The Streamlit application provides an interactive interface for entering customer information.

Users can:

1. Enter customer details.
2. Select Logistic Regression or Deep Neural Network.
3. Submit the information.
4. View the predicted churn probability.
5. View the final churn-risk classification.

The Streamlit application does not train the models. Instead, it sends customer data to the FastAPI backend and displays the API response.

## Project Architecture

```text
                  User
                   │
                   ▼
            Streamlit UI
               (app.py)
                   │
                   │ HTTP POST
                   ▼
             FastAPI API
               (api.py)
                   │
                   ▼
          Preprocessing Pipeline
                   │
          ┌────────┴────────┐
          ▼                 ▼
 Logistic Regression       DNN
          │                 │
          └────────┬────────┘
                   ▼
          Churn Probability
                   │
                   ▼
           High Risk / Low Risk
                   │
                   ▼
             Streamlit UI
```

## Project Structure

```text
Customer_Churn_Prediction/
│
├── app.py                         # Streamlit frontend
├── api.py                         # FastAPI backend
├── train.py                       # Model training and evaluation
├── requirements.txt               # Project dependencies
├── README.md
├── .gitignore
├── run_app.bat
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   ├── churn_dnn.keras
│   └── preprocessor.pkl
│
└── reports/
    ├── model_comparison.csv
    ├── classification reports
    ├── confusion matrices
    ├── ROC curves
    └── DNN learning curves
```

## How to Run

### 1. Create and activate the environment

Using Anaconda:

```bash
conda create -n churn_api python=3.12
conda activate churn_api
```

### 2. Navigate to the project

```bash
cd "C:\Users\Lakshmi P\OneDrive\Documents\Customer_Churn_Prediction"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python train.py
```

This creates the trained models and evaluation reports.

### 5. Start the FastAPI backend

Open an Anaconda Prompt:

```bash
conda activate churn_api
cd "C:\Users\Lakshmi P\OneDrive\Documents\Customer_Churn_Prediction"
uvicorn api:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit frontend

Open a second Anaconda Prompt:

```bash
conda activate churn_api
cd "C:\Users\Lakshmi P\OneDrive\Documents\Customer_Churn_Prediction"
streamlit run app.py
```

The Streamlit application will run at:

```text
http://localhost:8501
```

Keep both the FastAPI and Streamlit terminals running while using the application.

## Exploratory Data Analysis

The notebook contains exploratory analysis of:

* Overall churn distribution
* Churn by contract type
* Churn by tenure
* Churn by monthly charges
* Churn by payment method
* Churn by internet service
* Churn by senior-citizen status
* Churn by online security
* Churn by technical support

Visualizations are created using Pandas and Matplotlib.

## Future Improvements

* Optimize the classification threshold based on business retention costs.
* Add explainable AI techniques such as SHAP.
* Track experiments and model versions.
* Monitor model performance after deployment.
* Containerize the FastAPI and Streamlit applications.
* Deploy the application to a cloud platform.
