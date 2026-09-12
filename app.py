import streamlit as st
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load Iris dataset
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets["class"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=50)

model.fit(X_train, y_train)


# User inputs
st.title("🌸 Iris Flower Classifier")

st.write("Enter the flower measurements below:")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)


# Make prediction
if st.button("Predict Species"):

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=X.columns
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Species: {prediction[0]}")
