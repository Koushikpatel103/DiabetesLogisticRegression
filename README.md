# Diabetes Prediction using Logistic Regression

## Project Overview

This project predicts whether a person is diabetic or non-diabetic using **Machine Learning (Logistic Regression)** and deploys the model using **Streamlit**.

The model is trained on the **Pima Indians Diabetes Dataset** and predicts diabetes based on medical attributes.

---

## Features

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Missing Value Handling
- Feature Scaling
- Logistic Regression Model
- Model Evaluation
- Confusion Matrix
- Interactive Streamlit Web App
- Data Visualization Dashboard

---

## Dataset

Dataset Used:

**Pima Indians Diabetes Dataset**

Features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:

- Outcome
    - 0 → No Diabetes
    - 1 → Diabetes

---

## Project Workflow

### 1. Data Collection

Load dataset:

```python
diabetes.csv
```

### 2. Data Preprocessing

Performed:

- Missing value handling
- Replacing invalid zero values
- Feature scaling using StandardScaler
- Train-Test Split

### 3. Exploratory Data Analysis

Visualizations used:

- Correlation Heatmap
- Diabetes Distribution Graph
- Glucose vs BMI Scatter Plot

### 4. Model Training

Algorithm Used:

```python
LogisticRegression()
```

Train-Test Split:

```python
80% Train
20% Test
```

### 5. Model Evaluation

Metrics Used:

- Accuracy
- Confusion Matrix
- Classification Report

Typical Model Performance:

| Metric | Value |
|--------|--------|
| Accuracy | 75–85% |
| Precision | ~0.75 |
| Recall | ~0.70 |

---

## Streamlit Application

The Streamlit app allows users to:

- Enter patient medical information
- Predict diabetes risk instantly
- View prediction probability
- Explore dataset visualizations

Example Input:

```text
Pregnancies = 2
Glucose = 130
BMI = 28
Age = 35
```

Prediction:

```text
Not Diabetic
```

---

## Project Structure

```text
DiabetesPrediction/
│
├── diabetes.csv
├── train_model.py
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## Installation

Clone repository:

```bash
git clone <repository-link>
```

Move into project directory:

```bash
cd DiabetesPrediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train Model

Run:

```bash
python train_model.py
```

Generated files:

```text
model.pkl
scaler.pkl
```

---

## Run Streamlit App

Start application:

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## Machine Learning Concepts Used

- Classification
- Logistic Regression
- Feature Scaling
- Model Evaluation
- Data Visualization
- Deployment

---

## Future Improvements

Possible enhancements:

- Random Forest Classifier
- XGBoost
- Hyperparameter Tuning
- Better UI Design
- Cloud Deployment
- Feature Importance Analysis

---

## Screenshots

Add screenshots of:

- Streamlit UI
- Prediction Output
- Graph Dashboard

---

## Author

Koushik Patel

---

## License

This project is for educational and learning purposes.
