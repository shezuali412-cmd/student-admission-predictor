
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/binary.csv")

# Select inputs and target
X = df[["gre", "gpa", "rank"]]
y = df["admit"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Get user input
print("\nUniversity Admission Predictor")
print("------------------------------")

gre = float(input("Enter GRE score (220-800): "))
gpa = float(input("Enter GPA (2.26-4.00): "))
rank = int(input("Enter university rank (1-4): "))

# Validate inputs
if not 220 <= gre <= 800:
    print("GRE score must be between 220 and 800.")
elif not 2.26 <= gpa <= 4.00:
    print("GPA must be between 2.26 and 4.00.")
elif rank not in [1, 2, 3, 4]:
    print("Rank must be 1, 2, 3, or 4.")
else:
    student = pd.DataFrame(
        [[gre, gpa, rank]],
        columns=["gre", "gpa", "rank"]
    )

    prediction = model.predict(student)[0]
    probability = model.predict_proba(student)[0][1]

    if prediction == 1:
        print("\nModel prediction: Admitted")
    else:
        print("\nModel prediction: Not Admitted")

    print(f"Estimated admission probability: {probability:.1%}")
    print("Note: This is an educational estimate, not a real admission decision.")
