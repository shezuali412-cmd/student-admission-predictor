
import pandas as pd

# Load the real dataset
df = pd.read_csv("data/binary.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show dataset size
print("\nDataset shape:")
print(df.shape)

# Show column names
print("\nColumn names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Show basic statistics
print("\nDataset statistics:")
print(df.describe())



import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)

# 1. Admission distribution
admission_counts = df["admit"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(
    ["Not Admitted", "Admitted"],
    [admission_counts.get(0, 0), admission_counts.get(1, 0)]
)
plt.title("Admission Distribution")
plt.xlabel("Admission Status")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("images/admission_distribution.png")
plt.show()

# 2. GRE score distribution
plt.figure(figsize=(7, 4))
plt.hist(df["gre"], bins=15, edgecolor="black")
plt.title("GRE Score Distribution")
plt.xlabel("GRE Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("images/gre_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. GPA distribution
plt.figure(figsize=(7, 4))
plt.hist(df["gpa"], bins=15, edgecolor="black")
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("images/gpa_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. GRE vs GPA
plt.figure(figsize=(7, 5))

for status, label in [(0, "Not Admitted"), (1, "Admitted")]:
    group = df[df["admit"] == status]
    plt.scatter(
        group["gre"],
        group["gpa"],
        label=label,
        alpha=0.7
    )

plt.title("GRE Score vs GPA")
plt.xlabel("GRE Score")
plt.ylabel("GPA")
plt.legend()
plt.tight_layout()
plt.savefig("images/gre_vs_gpa.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Admission rate by university rank
admission_rate = df.groupby("rank")["admit"].mean() * 100

plt.figure(figsize=(7, 4))
plt.bar(admission_rate.index.astype(str), admission_rate.values)
plt.title("Admission Rate by University Rank")
plt.xlabel("University Rank")
plt.ylabel("Admission Rate (%)")
plt.tight_layout()
plt.savefig("images/admission_by_rank.png", dpi=300, bbox_inches="tight")
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Features and target
X = df[["gre", "gpa", "rank"]]
y = df["admit"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "ANN (MLP)": Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=3000,
        early_stopping=False,
        random_state=42
        ))
    ])
}

# Train and evaluate
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

    print(f"\n{'=' * 40}")
    print(name)
    print("=" * 40)
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not Admitted", "Admitted"],
        zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Not Admitted", "Admitted"],
        cmap="Blues"
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Compare models
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 5))

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

