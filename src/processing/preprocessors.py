import pandas as pd
from pandas.api import types as pandas_types
from src.utils.transformer import Transformer


class DropNa(Transformer):
    """Delete Missing value
    """

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.columns = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit statement to accomodate the sklearn pipeline."""

        X = self.transform(X)
        return X

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        X = pd.concat([X, pd.DataFrame(y)], axis=1)

        X = X.dropna()
        return X


class OneHotEncode(Transformer):
    """ Get dummies categorical variables
    """

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.categories = {}

    def fit_transform(self, X) -> pd.DataFrame:
        """

        """
        for var in self.variables:
            self.categories[var] = sorted(X[var].value_counts(dropna=True).index)
        X = self.transform(X)

        return X

    def transform(self, X) -> pd.DataFrame:
        """

        """
        for var in self.variables:
            X[var] = X[var].astype(
                pandas_types.CategoricalDtype(categories=self.categories[var])
            )

        X = pd.get_dummies(X)

        return X


class ObjectTypeToCategory(Transformer):

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit_transform(self, df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:

        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        obj_list_to_one_hot_encode = list(df.select_dtypes(include=['object']).columns)

        for col in obj_list_to_one_hot_encode:
            df[col] = df[col].astype('category')

        return df


class RemapTarget(Transformer):

    def __init__(self, target: str = None, mapping: dict = None) -> None:
        if mapping is None:
            raise Exception("Must contains mapping field and target")

        self.target = target
        self.mapping = mapping

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.target[0]] = df[self.target[0]].map(self.mapping)

        return df
