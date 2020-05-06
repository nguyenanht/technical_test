import pandas as pd


class Transformer:
    def __init__(self) -> None:
        pass

    def fit_transform(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> None:
        raise NotImplementedError
