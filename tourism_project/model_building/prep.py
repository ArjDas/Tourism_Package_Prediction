
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the registered dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Remove unnecessary columns
df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore", inplace=True)

# NOTE: Categorical columns are intentionally kept as raw strings.
# The training pipeline will handle encoding, and the deployment
# application must also provide raw categorical values.
# This ensures consistent representations during training and prediction.

# Separate independent and dependent variables
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Stratified train-test split maintains the purchase ratio
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save training and testing datasets
Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Training shape:", Xtrain.shape)
print("Testing shape:", Xtest.shape)
print("Purchase distribution:")
print(y.value_counts())
print("Gender values kept as:", sorted(X["Gender"].dropna().unique()))
