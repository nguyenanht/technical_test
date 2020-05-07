import logging

from sklearn.model_selection import train_test_split
import pandas as pd

import sys
sys.path.append('../')

# Custom import
from src.models.lightgbm_classifier import LgbClassifier
from src.extraction.data_management import Data

from src.processing import preprocessors
from src.processing.pipeline_management import save_pipeline
from src.tuning.bayesian_optimization import BayesOpt
from src.utils.pipeline import Pipeline
from src import __version__ as _version, config

_logger = logging.getLogger(__name__)

debug = False


def run_training() -> None:
    """Train the model."""

    # Read Training set
    # ================================================
    data_mngmnt = Data()
    data_mngmnt.format_excel_to_csv(filename=config.EXCEL_FILE)
    data_mngmnt.from_csv(config.DATA_FILE, sep=',')
    data = data_mngmnt.df

    # Divide dataset
    # ================================================
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.2,
        random_state=123,
        stratify=data[config.TARGET]
    )

    # DATA Pipeline
    # ================================================
    # Features
    transformers_features = [

        preprocessors.ObjectTypeToCategory(),
    ]
    preprocessing_pipeline = Pipeline(transformers=transformers_features)

    # TARGET, We remap the target between 0 and 1
    transformers_target = [
        preprocessors.RemapTarget(target=config.TARGET, mapping=config.MAPPING_TARGET)
    ]
    preprocessing_target_pipeline = Pipeline(transformers=transformers_target)

    # Fit transform X_train
    X_train = preprocessing_pipeline.fit_transform(X_train[config.FEATURES])
    X_test = preprocessing_pipeline.transform(X_test[config.FEATURES])
    # Fit transfrom y_train
    y_train = preprocessing_target_pipeline.fit_transform(y_train[config.TARGET])
    y_test = preprocessing_target_pipeline.transform(y_test[config.TARGET])

    # Save Pipeline model
    save_pipeline(pipeline_to_persist=preprocessing_pipeline,
                  pipeline_name=config.PIPELINE_PREPROCESSING_FEATURES_NAME)
    save_pipeline(pipeline_to_persist=preprocessing_pipeline,
                  pipeline_name=config.PIPELINE_PREPROCESSING_TARGET_NAME)
    _logger.info(f"saving model version: {_version}")

    # Save processed dataset
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    data_mngmnt.save_dataset(train, config.TRAINING_DATA_FILE, sep=',')
    data_mngmnt.save_dataset(test, config.TESTING_DATA_FILE, sep=',')

    del data_mngmnt  # clear variable in memory
    del data

    del train
    del test
    del preprocessing_pipeline
    del preprocessing_target_pipeline

    # if debug:
    #     make_prediction(input_data=data[config.FEATURES])
    # else:

    # Create Model
    # =================================================
    lgb = LgbClassifier(tag='training')  # set tag training to indicate to load dataset

    # Find best params for LightGbm model with Bayesian Optimization
    # ================================================
    optim = BayesOpt(lgb, lgb.bayes_evaluate, lgb.params)
    optim.tune(init_points=5, n_iter=20)
    lgb.set_best_params(optim.best_params)

    # Check for the robustness
    lgb.evaluate()

    # Finally fit with the best params
    lgb.fit()

    # Save model for further prediction in Flask
    lgb.save_model(config.MODEL_NAME)

    # Save metrics
    lgb.save_metrics(config.MODEL_NAME)


if __name__ == "__main__":
    run_training()
