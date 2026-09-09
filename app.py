"""Streamlit frontend for the Customer Churn Prediction API."""

import requests
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)


# --------------------------------------------------
# API configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def yes_no(label: str, default: str = "No") -> str:
    return st.selectbox(
        label,
        ["Yes", "No"],
        index=0 if default == "Yes" else 1
    )


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("Customer Churn Prediction")
st.caption(
    "Predict customer churn using Logistic Regression "
    "or Deep Neural Network through a FastAPI backend."
)


# --------------------------------------------------
# Check API status
# --------------------------------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    if health_response.status_code == 200:
        st.success("FastAPI backend is connected.")
    else:
        st.warning("FastAPI backend is not responding correctly.")

except requests.exceptions.RequestException:
    st.error(
        "FastAPI backend is not running. "
        "Please start the API using Uvicorn first."
    )


# --------------------------------------------------
# Customer input form
# --------------------------------------------------

with st.form("customer_form"):

    st.subheader("1. Customer Information")

    first, second, third, fourth = st.columns(4)

    with first:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda value:
                "Yes" if value else "No"
        )

    with second:
        partner = yes_no("Partner")
        dependents = yes_no("Dependents")

    with third:
        tenure = st.slider(
            "Tenure (months)",
            0,
            72,
            12
        )

    with fourth:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )


    # --------------------------------------------------
    # Services
    # --------------------------------------------------

    st.subheader("2. Services")

    first, second, third = st.columns(3)

    with first:

        phone_service = yes_no(
            "Phone Service",
            "Yes"
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with second:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with third:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    # --------------------------------------------------
    # Contract and billing
    # --------------------------------------------------

    st.subheader("3. Contract & Billing")

    first, second, third = st.columns(3)

    with first:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with second:

        paperless_billing = yes_no(
            "Paperless Billing",
            "Yes"
        )

    with third:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------

    st.subheader("4. Model Selection")

    selected_model = st.radio(
        "Select Prediction Model",
        [
            "Logistic Regression",
            "Deep Neural Network"
        ],
        horizontal=True
    )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )


# --------------------------------------------------
# Send prediction request to FastAPI
# --------------------------------------------------

if submitted:

    if selected_model == "Logistic Regression":
        model_name = "logistic_regression"
    else:
        model_name = "dnn"


    # Create request data
    customer_data = {
        "model": model_name,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }


    try:

        # Send request to FastAPI
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )


        # --------------------------------------------------
        # Successful response
        # --------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            probability = result["churn_probability"]
            prediction = result["prediction"]


            st.subheader("5. Prediction Result")


            if probability >= 0.5:

                st.error(
                    f"Prediction: {prediction}"
                )

                st.write(
                    "This customer shows a relatively high "
                    "probability of churn based on the selected model."
                )

            else:

                st.success(
                    f"Prediction: {prediction}"
                )

                st.write(
                    "This customer shows a relatively low "
                    "probability of churn based on the selected model."
                )


            st.metric(
                "Churn Probability",
                f"{probability:.1%}"
            )


            st.info(
                f"Prediction generated using: {result['model']}"
            )


        # --------------------------------------------------
        # Validation error
        # --------------------------------------------------

        elif response.status_code == 422:

            st.error(
                "Invalid customer input. "
                "Please check the entered values."
            )

            st.json(response.json())


        # --------------------------------------------------
        # Other API errors
        # --------------------------------------------------

        else:

            st.error(
                f"API request failed with status code "
                f"{response.status_code}"
            )

            st.json(response.json())


    except requests.exceptions.RequestException:

        st.error(
            "Could not connect to the FastAPI backend. "
            "Make sure Uvicorn is running."
        )