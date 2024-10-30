from load_data import import_data
from tidy_features import engineer_features
import os

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
print(train_w_features.shape)
print(train_w_features.columns)
print(train_w_features.nunique())
print(train_w_features.info())

# with_dummies = pd.get_dummies(
#     import_train,
#     columns=[

#     ]
# )
