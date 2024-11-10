import pandas as pd
from src.load_data import import_data
from src.preprocessing import full_preprocessing
import os
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    ValidationCurveDisplay
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

home = os.environ["PWD"]
print(home)

raw_train_df = import_data(
    os.path.join(
        home,
        "data",
        "train.csv"
    )
)

raw_external_test = import_data(
    os.path.join(
        home,
        "data",
        "test.csv"
    )
)

# Preprocessing
preprocess_train = full_preprocessing(raw_train_df)
preprocess_train_ids = preprocess_train["PassengerId"]
preprocess_train = preprocess_train.drop(columns = "PassengerId")
preprocess_ext_test = full_preprocessing(
    raw_external_test).assign(Embarked_unknown=0)
preprocess_ext_test_ids = preprocess_ext_test["PassengerId"]
preprocess_ext_test = preprocess_ext_test.drop(columns = "PassengerId")
# Split into train and test
X = preprocess_train.drop(
    columns=["Survived"]
)
y = preprocess_train.Survived

# Check columns are the same between train and the external test set after preprocessing
print(
    f"All columns equal between X and external test after preprocessing: {
        all(
            X.columns.sort_values() == preprocess_ext_test.columns.sort_values()
        )
    }"
)
# Reorder columns to match training input
preprocess_ext_test = preprocess_ext_test[X.columns]
all(X.columns == preprocess_ext_test.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.2
)

rf_param_grid = [
    {
        "scaling": [None, StandardScaler(), MinMaxScaler()],
        "rf__n_estimators": [10, 20, 50, 100, 200],
        "rf__max_depth": [2, 4, 8, 16, 32, 48, 96, None],
        "rf__min_samples_split": [2, 3, 4],
        "rf__min_samples_leaf": [1, 2, 3]
    }
]
pl = Pipeline(
    [
        ("scaling", "passthrough"),
        ("rf", RandomForestClassifier(
            random_state=42,
            verbose=0
        ))
    ]
)

grid = GridSearchCV(
    pl,
    cv=5,
    param_grid=rf_param_grid,
    n_jobs=-1,
    verbose=2,
    scoring="balanced_accuracy"
)
grid.fit(X_train, y_train)
acc = balanced_accuracy_score(grid.predict(X_test), y_test)
print(acc)

# Get predictions on external test set
external_pred = grid.predict(preprocess_ext_test)

ext_results = pd.DataFrame(preprocess_ext_test_ids).assign(Survived=external_pred)