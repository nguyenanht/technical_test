import pandas as pd
from src.config import config
import logging


_logger = logging.getLogger(__name__)


class Data:

    def __init__(self, dataframe=None):
        self.df = dataframe
        pass

    def from_csv(self, csv_file, sep=','):
        """Import data from csv format"""

        file_path_to_load = f"{config.DATASET_DIR}/{csv_file}"
        self.df = pd.read_csv(file_path_to_load, sep=',', low_memory=False, encoding="utf-8")

        return self

    @staticmethod
    def format_excel_to_csv(filename: str) -> None:
        """If excel is not readable with pandas because of comma separator,
        this function reformat and convert to a csv file.

        """

        # Excel file to reformat
        filepath = f"{config.DATASET_DIR}/{filename}"
        xl = pd.read_excel(filepath, header=None)
        xl['splitted'] = xl[0].str.split(',')

        # Nex dataframe formated
        df = pd.DataFrame()
        for i, elmt in enumerate(xl['splitted'][0]):
            df[elmt] = xl[1:]['splitted'].apply(lambda x: x[i])
        df = df.reset_index(drop=True)

        file_path_to_save = f"{config.DATASET_DIR}/{config.DATA_FILE}"
        df.to_csv(file_path_to_save, index=False, encoding='utf-8', sep=',')

    @staticmethod
    def save_dataset(df: pd.DataFrame, filename: str, sep) -> None:
        """Save dataset in format CSV
        """

        file_path_to_save = f"{config.DATASET_DIR}/{filename}"
        df.to_csv(file_path_to_save, sep=sep, index=None)
