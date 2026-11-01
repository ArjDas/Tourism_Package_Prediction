
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Remove unnecessary CustomerID column
df.drop(columns=["CustomerID"], inplace=True)

# NOTE: Categorical columns are intentionally left as raw strings.
# The training pipeline will handle categorical encoding.
# This ensures that training and serving use the same
# data representation.

# Separate independent and dependent variables
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the target class distribution consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save train-test splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Gender values kept as:", sorted(X["Gender"].unique()))
print("Marital Status values kept as:", sorted(X["MaritalStatus"].unique()))

print("Training data shape:", Xtrain.shape)
print("Testing data shape:", Xtest.shape)
