import pandas as pd
from src import config
from src import __version__ as _version
import joblib


class Model:
    def __init__(self) -> None:
        pass

    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> any:
        raise NotImplementedError

    def evaluate(self, params):
        raise NotImplementedError

    def save_model(self, filename: str, obj, timestamp) -> None:
        save_file_name = f"{filename}{_version}-{timestamp}"
        save_path = config.TRAINED_MODEL_DIR / save_file_name

        joblib.dump(obj, save_path)

