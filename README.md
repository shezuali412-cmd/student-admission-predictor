# University Admission Prediction & Analysis

## About the Project

This project explores student admission prediction using machine learning. It analyzes GRE scores, GPA, and university rank to predict admission outcomes.

The project compares three classification models:
- Logistic Regression
- Random Forest
- Artificial Neural Network (ANN) using MLPClassifier

## Dataset

The project uses a historical UCLA graduate admissions dataset containing 400 records.

**Features**
- `gre`: GRE score
- `gpa`: Grade Point Average
- `rank`: Applicant's university rank category

**Target**
- `admit`: 1 = Admitted, 0 = Not Admitted

Dataset source: https://stats.idre.ucla.edu/stat/data/binary.csv

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Machine Learning
- Artificial Neural Networks

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 73.8% | 83.3% | 20.0% | 32.3% |
| Random Forest | 63.8% | 42.9% | 48.0% | 45.3% |
| ANN (MLP) | 72.5% | 71.4% | 20.0% | 31.2% |

These results are from one train-test split and are not a guarantee of real-world admission outcomes.

## Project Features

- Dataset exploration and summary statistics
- Missing-value checks
- Data visualization
- Training and testing multiple classification models
- Evaluation using accuracy, precision, recall, and F1 score
- Confusion matrices and model comparison charts

## Important Note

This is an educational machine learning project using historical data. It should not be used to make real admission decisions.

## How to Run

1. Install Python.
2. Install the required libraries:

   `pip install pandas matplotlib scikit-learn`

3. Run the project:

   `python train.py`

## Author

Shehzad Ali — Artificial Intelligence Student
## Data Visualizations

### 1. Admission Distribution
![Admission Distribution](images/admission_distribution.png)

### 2. GRE Score Distribution
![GRE Score Distribution](images/gre_distribution.png)

### 3. GPA Distribution
![GPA Distribution](images/gpa_distribution.png)

### 4. GRE Score vs GPA
![GRE vs GPA](images/gre_vs_gpa.png)

### 5. Admission Rate by University Rank
![Admission by University Rank](images/admission_by_rank.png)