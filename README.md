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
3. requirements.txt file is located in docker directory

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
- If the model is missing, the script initializes a placeholder model.

### 3. `data/`

- Contains customer data (`internet_service_churn.csv`) used for analytics.
- Data is used for visualizing trends in customer churn.

### 4. `notebooks/`

- Jupyter Notebooks for data analysis and model training.

### 5. `docker/`

- Contains Docker-related configuration files to containerize and deploy the application.
- `docker-compose.yml` defines the services for running the app in a Docker container.

## Usage

1. **Start the application** (via Streamlit or Docker).
2. **Enter customer details** in the web interface.
3. **Run the prediction** to check churn probability.
4. **View recommendations** to retain at-risk customers.
5. **Analyze trends** using the provided analytics dashboard.
