
def engineer_features(df):
    func_df = df.assign(
        name_length=[len(x) for x in df.Name]
    )
    func_df = func_df.assign(
        age_unknown=func_df.Age.isna().astype(int)
    )
    func_df = func_df.drop(
        columns=[
            "PassengerId",
            "Name"
        ],
    )
    func_df.Age = func_df.Age.fillna(
        func_df.Age.mean()
    )
    func_df.Cabin = func_df.Cabin.fillna("unknown")
    func_df.Embarked = func_df.Embarked.fillna("unknown")
    return (func_df)
