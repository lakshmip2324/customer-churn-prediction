"""REST API for Customer Churn Prediction."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel, Field


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

preprocessor = joblib.load(
    MODELS_DIR / "preprocessor.pkl"
)

logistic_model = joblib.load(
    MODELS_DIR / "logistic_regression.pkl"
)

dnn_model = tf.keras.models.load_model(
    MODELS_DIR / "churn_dnn.keras"
)


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "REST API for predicting customer churn "
        "using Machine Learning and Deep Learning models."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Input validation model
# --------------------------------------------------

class CustomerData(BaseModel):

    # Model selection
    model: Literal["logistic_regression", "dnn"]

    # Customer information
    gender: Literal["Male", "Female"]

    SeniorCitizen: int = Field(
        ge=0,
        le=1
    )

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    # Account information
    tenure: int = Field(
        ge=0,
        le=72
    )

    # Phone service
    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    # Internet service
    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    # Contract and billing
    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal[
        "Yes",
        "No"
    ]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    # Charges
    MonthlyCharges: float = Field(
        ge=0
    )

    TotalCharges: float = Field(
        ge=0
    )


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get(
    "/health",
    tags=["Health"]
)
def health_check():

    return {
        "status": "healthy",
        "message": "Customer Churn Prediction API is running"
    }


# --------------------------------------------------
# Churn prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    tags=["Prediction"]
)
def predict_churn(customer: CustomerData):

    # Convert Pydantic object to dictionary
    customer_data = customer.model_dump()

    # Get selected model
    selected_model = customer_data.pop("model")

    # Convert customer data into DataFrame
    customer_df = pd.DataFrame(
        [customer_data]
    )

    # Apply the same preprocessing used during training
    transformed_customer = preprocessor.transform(
        customer_df
    )

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    if selected_model == "logistic_regression":

        probability = float(
            logistic_model
            .predict_proba(transformed_customer)[:, 1][0]
        )

    # --------------------------------------------------
    # Deep Neural Network
    # --------------------------------------------------

    elif selected_model == "dnn":

        # Convert sparse matrix to dense matrix if required
        dnn_input = (
            transformed_customer.toarray()
            if hasattr(transformed_customer, "toarray")
            else transformed_customer
        )

        probability = float(
            dnn_model
            .predict(
                dnn_input,
                verbose=0
            )[0][0]
        )

    # --------------------------------------------------
    # Convert probability into prediction
    # --------------------------------------------------

    if probability >= 0.5:

        prediction = "High Risk of Churn"

    else:

        prediction = "Low Risk of Churn"


    # --------------------------------------------------
    # API response
    # --------------------------------------------------

    return {
        "model": selected_model,
        "churn_probability": round(
            probability,
            4
        ),
        "prediction": prediction
    }
