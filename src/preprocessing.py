import pandas as pd


def engineer_features(df):
    func_df = df.assign(
        name_length=[len(x) for x in df.Name]
    )
    func_df = func_df.assign(
        age_unknown=func_df.Age.isna().astype(int)
    )
    func_df = func_df.drop(
        columns=[
            "Name",
            "Cabin",
            "Ticket"
        ],
    )
    func_df.Age = func_df.Age.fillna(
        func_df.Age.mean()
    )
    func_df.Embarked = func_df.Embarked.fillna("unknown")
    return (func_df)


def create_dummies(df):
    func_df = pd.get_dummies(
        df,
        columns=[
            "Embarked",
            "Sex"
        ],
        drop_first=True,
        dtype=int
    )
    return func_df


def full_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """wrapper around all preprocessing steps"""
    func_df = (
        df
        .pipe(engineer_features)
        .pipe(create_dummies)
    )
    return func_df
