import mlflow, pandas as pd, random, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:50000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Latihan Credit Scoring")

data = pd.read_csv("train_pca.csv")

# Change int to float
data = data.apply(lambda col: col.astype(float) if col.dtype == 'int64' else col)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)

input_example = X_train.iloc[0:5]

with mlflow.start_run():
    # Log parameters
    n_estimators = 505
    max_depth = 37
    
    # Enable auto logging
    mlflow.autolog()
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Log model explicitly
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
      
    # Log metrics
    accuracy = model.score(X_test, y_test)
    
    mlflow.log_metric("accuracy", accuracy)
