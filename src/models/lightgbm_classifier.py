from sklearn.model_selection import StratifiedKFold
import joblib
import pandas as pd
import numpy as np
from os import path, listdir
from os.path import isfile, join

import lightgbm as lgb
import csv
from datetime import datetime

from src.config import config
from src.metrics.cust_metric import get_auc_score, find_optimal_cutoff, get_accuracy_score
from src.tuning.utils import find_optimal_cutoff_auc
from src.utils.model import Model
from src import __version__ as _version


class LgbClassifier(Model):
    """Model lightgbm
    """

    def __init__(self, tag=None):

        if tag not in [None, 'training']:
            raise Exception("tag must be None or 'training'")

        self.model = None
        self.threshold = 0.5  # For future prediction
        self.auc_score = None
        self.acc_score = None
        self.timestamp = None

        # Parameter of the model
        # ==============================
        self.params = {
            'num_leaves': (2, 10),
            'feature_fraction': (0.6, 1.0),
            'bagging_fraction': (0.6, 1.0),
            'bagging_freq': (5, 10),
            "max_depth": (2, 8),
        }

        # Defined splitter of dataset for evaluate and bayes_evaluate function
        # ==============================
        self.kfold = 5
        self.skf = StratifiedKFold(n_splits=self.kfold, random_state=42, shuffle=True)

        # Variable
        # ==============================
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        if tag == "training":
            self.load_data()

    def load_data(self):
        # Load processed DataFrame
        # ==============================
        train = pd.read_csv(f"{config.DATASET_DIR}/{config.TRAINING_DATA_FILE}", sep=",")
        test = pd.read_csv(f"{config.DATASET_DIR}/{config.TESTING_DATA_FILE}", sep=",")

        # get columns name
        features_col = [col for col in train.columns if col not in config.TARGET]
        target_col = config.TARGET[0]

        # Dataset
        obj_list_to_one_hot_encode = list(train.select_dtypes(include=['object']).columns)
        for col in obj_list_to_one_hot_encode:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')

        self.X_train = train[features_col]
        self.y_train = train[target_col]
        self.X_test = test[features_col]
        self.y_test = test[target_col]

    def fit(self):
        """ Fit all the dataset
        """
        X = pd.concat([self.X_train, self.X_test], axis=0)
        y = pd.concat([self.y_train, self.y_test], axis=0)

        # Transform obj to category required by lightgbm
        obj_list_to_one_hot_encode = list(X.select_dtypes(include=['object']).columns)
        for col in obj_list_to_one_hot_encode:
            X[col] = X[col].astype('category')

        train_data = lgb.Dataset(data=X,
                                 label=y,
                                 weight=10 ** y)

        self.model = lgb.train(self.params, train_data, verbose_eval=False)

        y_pred = self.model.predict(X)

        # Find optimal probability threshold
        self.threshold = find_optimal_cutoff_auc(y, y_pred)

        # current date and time
        now = datetime.now()
        # Set timestamp for saving model and metrics
        self.timestamp = datetime.timestamp(now)

    def predict(self, X):
        result = self.model.predict(X)
        pred_optimized = (result > self.threshold).astype(int)
        return pred_optimized

    def evaluate(self):
        """Generate metrics
        todo :
            check the shift between previous model
            to authorize this model to go on production

        """
        train_data = lgb.Dataset(data=self.X_train,
                                 label=self.y_train,
                                 weight=10 ** self.y_train)

        optimize_model = lgb.train(
            self.params,
            train_data,
            verbose_eval=False
        )

        y_pred = optimize_model.predict(self.X_test)

        self.threshold = find_optimal_cutoff(self.y_test, y_pred)
        pred_opt = (y_pred > self.threshold).astype(int)  # Find prediction to the dataframe applying threshold

        self.auc_score = round(get_auc_score(self.y_test, y_pred), 3)
        self.acc_score = round(get_accuracy_score(self.y_test, pred_opt), 3)

        print('roc auc lightgbm : ', str(self.auc_score))
        print('accuracy lightgbm : ', str(self.acc_score))

    def bayes_evaluate(self, num_leaves, feature_fraction, bagging_fraction, bagging_freq, max_depth):
        """Compute the best parameters of XGboost

        Parameters
        --------
            max_depth:
                It describes the maximum depth of tree. This parameter is used to handle model overfitting.

            bagging_freq:
                every tree is learning on a different subsample of your dataset

            bagging_fraction:
                specifies the fraction of data to be used for each iteration
                and is generally used to speed up the training and avoid overfitting.

            feature_fraction:
                LightGBM will select 80% of parameters randomly in each iteration for building trees

            num_leaves:
                number of leaves in full tree

        Returns
        --------
            auc score
        """

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': int(num_leaves),
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': int(bagging_freq),
            'is_unbalance': 'true',
            'learning_rate': 0.001,
            'max_depth': int(max_depth),
            'verbose': -1
        }

        train_data = lgb.Dataset(data=self.X_train,
                                 label=self.y_train,
                                 weight=10 ** self.y_train)

        cv_result = lgb.cv(
            params,
            train_data,
            num_boost_round=150,
            nfold=5,
            metrics='auc',
            early_stopping_rounds=30
        )

        return np.array(cv_result['auc-mean']).max()

    def set_best_params(self, params: dict) -> None:
        # Best parameters
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': int(params['num_leaves']),
            'feature_fraction': params['feature_fraction'],
            'bagging_fraction': params['bagging_fraction'],
            'bagging_freq': int(params['bagging_freq']),
            'is_unbalance': 'true',
            'learning_rate': 0.001,
            'max_depth': int(params['max_depth']),
            'verbose': -1
        }

    def save_model(self, filename: str) -> None:
        """Save the prediction Model"""

        model = {
            "model": self.model,
            "threshold": self.threshold,
            "auc_score": self.auc_score,
            "acc_score": self.acc_score
        }

        super().save_model(filename, model, self.timestamp)  # inherit of parent class Model

    def load_model(self, filename: str = None) -> None:
        """Load Model or intermediate model.
        By default, without filename, it loads de the last model based on timestamp
        """

        if filename is None:
            files = [f for f in listdir(config.TRAINED_MODEL_DIR) if isfile(join(config.TRAINED_MODEL_DIR, f))]
            # Get the index file with the maximum timestamp
            idx_most_recent_models = files.index(max([f.strip('-') for f in files]))
            filepath = config.TRAINED_MODEL_DIR / files[idx_most_recent_models]
        else:
            filepath = config.TRAINED_MODEL_DIR / filename

        trained_model = joblib.load(filename=filepath)
        #
        self.model = trained_model['model']
        self.threshold = trained_model['threshold']
        self.auc_score = trained_model['auc_score']
        self.acc_score = trained_model['acc_score']

    def save_metrics(self, model_name:str) -> None:
        model_name = f"{model_name}{_version}-{self.timestamp}"
        file_path = config.TRAINED_MODEL_METRICS_DIR / config.METRICS_FILE
        if path.exists(file_path):

            fields = [model_name, self.acc_score, self.auc_score]
            with open(file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        else:

            header = ['model_version', 'accuracy_score', 'auc_score']
            fields = [model_name, self.acc_score, self.auc_score]
            with open(file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(fields)

