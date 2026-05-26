import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("diabetes.csv")

# Title
st.markdown(
    """
    <h1 style='text-align:center;color:#2E86C1'>
    🩺 Diabetes Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.write(
    "Predict whether a patient is diabetic using Logistic Regression."
)

# Sidebar
st.sidebar.header("Patient Information")

Pregnancies = st.sidebar.number_input(
    "Pregnancies", 0, 20, 1
)

Glucose = st.sidebar.number_input(
    "Glucose", 0, 300, 120
)

BloodPressure = st.sidebar.number_input(
    "Blood Pressure", 0, 200, 70
)

SkinThickness = st.sidebar.number_input(
    "Skin Thickness", 0, 100, 20
)

Insulin = st.sidebar.number_input(
    "Insulin", 0, 900, 80
)

BMI = st.sidebar.number_input(
    "BMI", 0.0, 70.0, 25.0
)

DiabetesPedigreeFunction = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    0.0, 3.0, 0.5
)

Age = st.sidebar.number_input(
    "Age", 1, 120, 30
)

# Main layout
col1, col2 = st.columns(2)

with col1:

    st.subheader("Patient Inputs")

    patient_data = pd.DataFrame({
        'Feature': [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DPF',
            'Age'
        ],
        'Value': [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]
    })

    st.dataframe(
        patient_data,
        use_container_width=True
    )

with col2:

    st.subheader("Prediction")

    if st.button("Predict"):

        data = np.array([
            [
                Pregnancies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DiabetesPedigreeFunction,
                Age
            ]
        ])

        data = scaler.transform(data)

        prediction = model.predict(data)
        probability = model.predict_proba(data)

        diabetic_prob = probability[0][1] * 100

        st.progress(
            int(diabetic_prob)
        )

        if prediction[0] == 1:

            st.error(
                f"High Risk of Diabetes\n\nProbability: {diabetic_prob:.2f}%"
            )

        else:

            st.success(
                f"Low Risk of Diabetes\n\nProbability: {100-diabetic_prob:.2f}%"
            )

        # Probability chart
        fig, ax = plt.subplots()

        labels = [
            "No Diabetes",
            "Diabetes"
        ]

        probs = probability[0]

        ax.bar(
            labels,
            probs
        )

        ax.set_ylabel(
            "Probability"
        )

        ax.set_title(
            "Prediction Probability"
        )

        st.pyplot(fig)

# Divider
st.markdown("---")

# Graph Section
st.header("Dataset Analysis")

graph1, graph2 = st.columns(2)

# Outcome Distribution
with graph1:

    st.subheader("Diabetes Distribution")

    fig1, ax1 = plt.subplots()

    df['Outcome'].value_counts().plot(
        kind='bar',
        ax=ax1
    )

    ax1.set_xlabel(
        "Outcome"
    )

    ax1.set_ylabel(
        "Count"
    )

    st.pyplot(fig1)

# Heatmap
with graph2:

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots(
        figsize=(7,5)
    )

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax2
    )

    st.pyplot(fig2)

# Feature Comparison
st.subheader("Glucose vs BMI")

fig3, ax3 = plt.subplots()

sns.scatterplot(
    data=df,
    x='Glucose',
    y='BMI',
    hue='Outcome',
    ax=ax3
)

st.pyplot(fig3)

# Footer
st.markdown("---")
st.caption(
    "Built with Streamlit + Logistic Regression"
)