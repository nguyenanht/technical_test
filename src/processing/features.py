import pandas as pd
from src.utils.transformer import Transformer


class TemporalTransformer(Transformer):
    """Date transformer to month, year, day."""

    def __init__(self, variables=None):
        super().__init__()
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = pd.to_datetime(X[feature])  # On spécifie le type datetime à la colonne date
            X['annee'] = X[feature].dt.year
            X['mois'] = X[feature].dt.month
            X['jour'] = X[feature].dt.weekday
            X = X.drop(feature, axis=1)

        return X



