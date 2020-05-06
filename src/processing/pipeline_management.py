import pandas as pd
import joblib
from src.utils.pipeline import Pipeline
from src.config import config
from src import __version__ as _version
import logging
import typing as t
from os import listdir
from os.path import isfile, join
from datetime import datetime


_logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str, sep=',') -> pd.DataFrame:
    """Load dataset in format CSV

    Parameters
    ----------
    file_name : str
        The filename to load
    sep : str
        separator of the csv
    Examples
    --------
    >>> from  src.processing.pipeline_management import load_dataset
    >>> df = load_dataset('datasets/test_technique.csv')
    """
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}", sep=sep)
    _data = _data[config.FEATURES + [config.TARGET]]
    return _data


def save_dataset(df: pd.DataFrame, filename: str, sep) -> None:
    """Save dataset in format CSV
    """

    file_path_to_save = f"{config.DATASET_DIR}/{filename}"
    df.to_csv(file_path_to_save, sep=sep, index=None)


def save_pipeline(*, pipeline_to_persist: str = None, pipeline_name: str = None) -> None:
    """Persist the pipeline.
    Saves the versioned model or pretreatment pipeline, and overwrites any previous
    saved intermediate_model. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.

    Parameters
    ----------
    pipeline_to_persist : Pipeline
        The Pipeline object to persist

    pipeline_name: Str
        Name of the output pipeline saved

    Examples
    --------
    >>> from src.utils.pipeline import Pipeline
    >>> from  src.processing.pipeline_management import save_pipeline
    >>> transformers = []
    >>> pipe = Pipeline(transformers=transformers)
    >>> save_pipeline(pipeline_to_persist=pipe, pipeline_name='pipeline_features')
    """

    # Exception
    if pipeline_to_persist is None:
        raise Exception("Must contains attribute type.")

    # current date and time
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    save_file_name = f"{pipeline_name}_output_v{_version}-{timestamp}"
    save_path = config.TRAINED_PIPELINE_DIR / pipeline_name /save_file_name

    # remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved model: {save_file_name}")


def load_pipeline(*, file_name: str = None) -> Pipeline:
    """Load a persisted pipeline.

    Parameters
    ----------
    file_name : str
        filename of Pipeline to load

    """

    if file_name is None:
        files_path = config.TRAINED_PIPELINE_DIR / 'preprocessing_features_pipeline'
        files = [f for f in listdir(files_path) if isfile(join(files_path, f))]
        # Last index of pipeline model
        idx_most_recent_models = files.index(max([f.strip('-') for f in files]))
        file_path = files_path / files[idx_most_recent_models]

    else:

        file_path = config.TRAINED_PIPELINE_DIR / 'preprocessing_features_pipeline' / file_name

    trained_model = joblib.load(filename=file_path)

    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.

    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_PIPELINE_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def move_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.

    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_PIPELINE_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()