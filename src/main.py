from load_data import import_data
from tidy_features import engineer_features, create_dummies
import os
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

home = os.environ["PWD"]
print(home)

raw_train_df = import_data(
    os.path.join(
        home,
        "data",
        "train.csv"
    )
)

print(raw_train_df.shape)

# features
train_w_features = engineer_features(raw_train_df)

# One-hot encoding
train_w_dummies = create_dummies(train_w_features)

X = train_w_dummies.drop(
    columns=["Survived"]
)
y = train_w_dummies.Survived

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.2
)
rf_param_grid = [
    {
        "rf__n_estimators": [10, 20, 50, 100],
        "rf__max_depth": [2, 5, 10, 100, None],
        "rf__min_samples_split": [2, 3, 4],
        "rf__min_samples_leaf": [1, 2, 3]
    }
]
pl = Pipeline(
    [
        ("scaling", StandardScaler()),
        ("rf", RandomForestClassifier(
            random_state=42,
            verbose=1
        ))
    ]
)

grid = GridSearchCV(
    pl,
    cv=5,
    param_grid=rf_param_grid,
    n_jobs=-1
)
grid.fit(X_train, y_train)
acc = accuracy_score(grid.predict(X_test), y_test)
print(acc)
