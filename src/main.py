from load_data import import_data
from tidy_features import engineer_features, create_dummies
import os
from sklearn.model_selection import {
    train_test_split,
    cross_val_score,
    GridSearchCV
}
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
        n_estimators: [10, 20, 50, 100],
        max_depth = [2, 5, 10, 100, None],
        min_samples_split = [2, 3, 4],
        min_samples_leaf = [1, 2, 3]
    }
]
pl = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        njobs=-1,
        random_state=42
    )
)

pl.fit(X_train, y_train)

pred = pl.predict(X_test)

acc = accuracy_score(pred, y_test)
print(acc)
