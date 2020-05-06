import pandas as pd
from typing import List
from src.utils.transformer import Transformer


class Pipeline(Transformer):
    """Pipeline for data"""
    def __init__(self, transformers: List[Transformer]) -> None:
        Transformer.__init__(self)
        self.transformers: List[Transformer] = transformers

    def fit_transform(self, df_to_fit: pd.DataFrame) -> pd.DataFrame:

        df: pd.DataFrame = df_to_fit.copy()

        for idx, tr in enumerate(self.transformers):
            df = tr.fit_transform(df)
            self.transformers[idx] = tr

        return df

    def transform(self, df_to_transform: pd.DataFrame) -> pd.DataFrame:
        # print('transform')
        df: pd.DataFrame = df_to_transform.copy()

        for tr in self.transformers:

            df = tr.transform(df)

        return df
