import pandas as pd
from src.models.lightgbm_classifier import LgbClassifier
from src.processing.pipeline_management import load_pipeline
from src.config import config
from src import __version__ as _version

from os import listdir
from os.path import isfile, join

import logging
import typing as t

_pipe = load_pipeline()
_model = LgbClassifier()
_model.load_model()

_logger = logging.getLogger(__name__)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict], id_model: str = None) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
    if isinstance(input_data, dict):
        data = pd.DataFrame(input_data, index=[0])
    else:
        data = pd.DataFrame(input_data)

    # Transform data
    transformed_data = _pipe.transform(data[config.FEATURES])

    # If we use specifif model, else use default model based on last timestamp
    if id_model is not None:
        specific_model = LgbClassifier()
        specific_model.load_model(filename=id_model)
        output = specific_model.predict(transformed_data)
    else:
        output = _model.predict(transformed_data)

    results = {"predictions": output, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {input_data} "
        f"Predictions: {results}"
    )

    return results


def get_models_list():
    """ Get list of model in trained_models/processed_model
    """

    files = [f for f in listdir(config.TRAINED_MODEL_DIR) if isfile(join(config.TRAINED_MODEL_DIR, f))]
    return files
