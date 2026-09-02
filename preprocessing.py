import pandas as pd
from sklearn.impute import SimpleImputer


class Preprocessing:
    def __init__(self, target=None, problem=None):
        self.target = target
        self.problem = problem

    def choose_target(self):
        return f"The target variable is: {self.target} and the problem type is: {self.problem}"

    def separate_features_target(self, data, target=None):
        if target is None:
            target = self.target
        X = data.drop(columns=[target])
        y = data[target]
        return X, y

    def get_feature_types(self, data):
        numeric = data.select_dtypes(include=['number']).columns.tolist()
        categorical = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        # ensure target is not returned as a feature
        numeric = [c for c in numeric if c != self.target]
        categorical = [c for c in categorical if c != self.target]
        return numeric, categorical

    def clean_data(self, data, numeric_strategy='median', categorical_strategy='most_frequent', drop_threshold=0.9):
        data = data.copy()

        # drop columns with too many missing values
        missing_frac = data.isnull().mean()
        cols_to_drop = missing_frac[missing_frac > drop_threshold].index.tolist()
        if cols_to_drop:
            data.drop(columns=cols_to_drop, inplace=True)

        num_cols, cat_cols = self.get_feature_types(data)

        if num_cols:
            num_imp = SimpleImputer(strategy=numeric_strategy)
            data[num_cols] = num_imp.fit_transform(data[num_cols])

        if cat_cols:
            cat_imp = SimpleImputer(strategy=categorical_strategy, fill_value='missing')
            data[cat_cols] = cat_imp.fit_transform(data[cat_cols])

        return data

    def __repr__(self):
        return f"Preprocessing(target={self.target}, problem={self.problem})"
