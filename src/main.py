import pandas as pd
import matplotlib.pyplot as plt
import os

home = os.environ["PWD"]

import_train = pd.read_csv(
    os.path.join(
        home,
        "data",
        "train.csv"
    )
)

# Add features
# name length
add_name_length = import_train.assign(
    name_length=[len(x) for x in import_train.Name]
)


# Drop unnecessary columns
drop_train = add_name_length.drop(
    labels=[
        "PassengerId",
        "Name"
    ],
    axis=1
)


with_dummies = pd.get_dummies(
    import_train,
    columns=[

    ]
)
