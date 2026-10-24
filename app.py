
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Page settings
st.set_page_config(
    page_title="University Admission Predictor",
    page_icon="🎓"
)

# Load dataset
df = pd.read_csv("binary.csv")

X = df[["gre", "gpa", "rank"]]
y = df["admit"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Create and train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Website heading
st.title("🎓 University Admission Predictor")

st.write(
    "Enter your academic information to get an "
    "educational admission prediction."
)

# User inputs
gre = st.number_input(
    "GRE Score", min_value=220, max_value=800, value=500
)

gpa = st.number_input(
    "GPA", min_value=2.26, max_value=4.00,
    value=3.00, step=0.01, format="%.2f"
)

rank = st.selectbox(
    "University Rank",
    options=[1, 2, 3, 4]
)

# Prediction button
if st.button("Predict Admission"):
    student = pd.DataFrame(
        [[gre, gpa, rank]],
        columns=["gre", "gpa", "rank"]
    )

    prediction = model.predict(student)[0]
    probability = model.predict_proba(student)[0][1]

    if prediction == 1:
        st.success("Model Prediction: Admitted")
    else:
        st.error("Model Prediction: Not Admitted")

    st.metric(
        "Estimated Admission Probability",
        f"{probability:.1%}"
    )

st.caption(
    "Educational project using historical data. "
    "This result is not a real admission decision."
)
