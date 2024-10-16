# Insurance Prediction Application

This is a Python-based application that predicts medical expenses for insurance using various machine learning models, including Linear Regression, Decision Tree Regressor, and XGBoost Regressor. The application also provides a user interface (UI) built with Tkinter for predicting medical expenses and determining if insurance is necessary based on a threshold.

## Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [UI Functionality](#ui-functionality)
- [Dataset](#dataset)
- [License](#license)

## Features

- Data analysis and visualization using Seaborn and Matplotlib.
- Implementation of Linear Regression, Decision Tree, and XGBoost models for predicting insurance expenses.
- Cross-validation of the models to compare performance based on Mean Squared Error (MSE).
- Tkinter-based GUI for interactive predictions of insurance expenses based on user input.
- Simple decision logic to determine if insurance can be provided based on predicted expenses.

## Technologies

- Python 3.x
- Libraries: 
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - XGBoost
  - Tkinter

## Installation

To get started with the application, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/insurance-prediction-app.git
   ```

2. Navigate to the project directory:

   ```bash
   cd insurance-prediction-app
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the `insurance prediction.csv` dataset is in the project directory. 

5. Run the application:

   ```bash
   python insurance_prediction.py
   ```

## Usage

### Running the Data Analysis

To explore the data and visualize trends in expenses, run the Python script `insurance_analysis.py`. This script provides insights into:

- Expense distribution based on gender, smoker status, age, BMI, etc.
- Visualization of categorical variables using bar plots and pie charts.
- Model training and evaluation using Linear Regression, Decision Tree, and XGBoost models.

### Tkinter UI for Predictions

The project includes a Tkinter-based graphical user interface for making predictions. Once launched, the application will prompt the user to enter their information:

- Age
- Sex
- BMI
- Number of children
- Smoker status
- Region

Upon clicking "Predict Expenses," the app will predict medical expenses and display a message indicating whether insurance is recommended based on the prediction.

## Models

Three models are trained and evaluated:

1. **Linear Regression**: A simple linear approach for predicting expenses.
2. **Decision Tree Regressor**: A tree-based model that splits data into nodes based on feature values.
3. **XGBoost Regressor**: An advanced boosting algorithm for higher accuracy.

The best model is chosen based on the lowest Mean Squared Error (MSE) using cross-validation.

## UI Functionality

The Tkinter UI enables users to enter personal details such as age, gender, BMI, smoker status, number of children, and region, and predict their medical expenses. The app compares the predicted expense against a threshold to decide if insurance is necessary.

## Dataset

The dataset `insurance prediction.csv` contains the following columns:

- `age`: Age of the applicant
- `sex`: Gender of the applicant
- `bmi`: Body Mass Index of the applicant
- `children`: Number of children the applicant has
- `smoker`: Whether the applicant smokes (yes/no)
- `region`: Applicant's region of residence
- `expenses`: Medical expenses incurred by the applicant

The dataset is preprocessed using one-hot encoding for categorical variables before being passed to the machine learning models.
