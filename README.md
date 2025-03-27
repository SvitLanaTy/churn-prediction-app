# Churn Prediction App Documentation

## Overview

The **Churn Prediction App** is a machine-learning-powered web application designed to predict customer churn in a telecommunications company. The application utilizes Streamlit for the front-end interface and a pre-trained machine learning model to make predictions based on customer data.

## Project Structure

```
churn-prediction-app/
│-- .git/                 # Git repository (hidden)
│-- data/                 # Data storage directory
│-- docker/               # Docker configuration files
│-- models/               # Directory for storing ML models
│-- notebooks/            # Jupyter notebooks for analysis
│-- app.py                # Main application script
│-- README.md             # Project overview and instructions
│-- .gitignore            # Git ignore rules

```

## Installation and Setup

### Running the Application using Docker

1. Navigate to the `docker` directory:
   ```sh
   cd docker
   ```
2. Run the application using Docker Compose:
   ```sh
   docker-compose -f docker-compose.yml up --build
   ```
3. After a successful launch, the application will be available at the address:
   ```sh
   http://localhost:49153
   ```

## Application Components

### 1. `app.py`

The main application script is built using **Streamlit**. Key functionalities include:

- User interface for entering customer data.
- Machine learning model loading and prediction.
- Data visualization using **Plotly**.
- Displaying churn prediction results and recommendations.

#### Features:

- **Customer Data Input:** Users can input service usage details, contract length, and service failures.
- **Prediction Model:** A trained XGBoost model (stored in `models/xgboost_model.pkl`) is used for prediction.
- **Visualization:** Displays churn probability, contributing factors, and analytics on customer churn.
- **Recommendations:** Provides insights on reducing customer churn risk.

### 2. `models/`

- Stores the trained machine learning model (`xgboost_model.pkl`).

### 3. `data/`

- Contains customer data (`internet_service_churn.csv`) used for analytics.
- Data is used for visualizing trends in customer churn.

### 4. `notebooks/`

- Jupyter Notebooks for data analysis and model training.

### 5. `docker/`

- Contains Docker-related configuration files to containerize and deploy the application.
- `docker-compose.yml` defines the services for running the app in a Docker container.

## Model Performance Metrics

Three machine learning models were trained and compared. The best-performing model (XGBoost) was selected for deployment.

### Model Comparison:

| Model              | Accuracy | Recall   | Precision | F1-score | Mean CV Accuracy |
| ------------------ | -------- | -------- | --------- | -------- | ---------------- |
| CatBoost           | 0.936991 | 0.936012 | 0.950869  | 0.943382 | 0.938276         |
| XGBoost            | 0.938521 | 0.935764 | 0.953741  | 0.944667 | 0.941214         |
| LogisticRegression | 0.874957 | 0.922495 | 0.863795  | 0.892180 | 0.870692         |

### Best Model (XGBoost) Performance:

| Metric           | Score    |
| ---------------- | -------- |
| Accuracy         | 0.938521 |
| Recall           | 0.935764 |
| Precision        | 0.953741 |
| F1-score         | 0.944667 |
| Mean CV Accuracy | 0.941214 |
| Overall Score    | 0.942782 |

## Usage

1. **Start the application** (via Streamlit or Docker).
2. **Enter customer details** in the web interface.
3. **Run the prediction** to check churn probability.
4. **View recommendations** to retain at-risk customers.
5. **Analyze trends** using the provided analytics dashboard.
