import joblib
import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import roc_curve, auc

from src.config import config
from sklearn.model_selection import StratifiedKFold
from src.tuning.utils import find_optimal_cutoff_auc
from src.utils.model import Model
from src import __version__ as _version


class XgbClassifier(Model):
    """Model xgboost
    """

    def __init__(self, tag=None):

        if tag not in [None, 'training']:
            raise Exception("tag must be None or 'training'")

        self.model = None
        self.threshold = 0.5  # For future prediction
        self.roc_auc = None
        self.num_boost_round = None

        # Parameter of the model
        # ==============================
        self.params = {
            'learning_rate': (0.01, 0.1),
            'min_child_weight': (1, 20),
            'gamma': (0, 5),
            'subsample': (0.8, 1),
            'colsample_bytree': (0.3, 0.8),
            'max_depth': (2, 8)
        }
        self.num_boost_round = None

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
        train = pd.read_csv(f"{config.DATASET_DIR}/{config.TRAINING_DATA_FILE}", sep=";")
        test = pd.read_csv(f"{config.DATASET_DIR}/{config.TESTING_DATA_FILE}", sep=";")

        # get columns name
        features_col = [col for col in train.columns if col not in config.TARGET]
        target_col = config.TARGET
        # Dataset
        self.X_train = train[features_col]
        self.y_train = train[target_col]
        self.X_test = test[features_col]
        self.y_test = test[target_col]

    def fit(self):
        X = pd.concat([self.X_train, self.X_test], axis=0)
        y = pd.concat([self.y_train, self.y_test], axis=0)
        d_train = xgboost.DMatrix(X, y)

        # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
        # and the custom metric (maximize=True tells xgb that higher metric is better)
        self.model = xgboost.train(self.params,
                                   d_train,
                                   num_boost_round=self.num_boost_round,
                                   verbose_eval=10)

        d_X = xgboost.DMatrix(X)
        y_pred = self.model.predict(d_X)

        # Find optimal probability threshold
        self.threshold = find_optimal_cutoff_auc(y, y_pred)

    def predict(self, X):
        d_X = xgboost.DMatrix(X)
        result = self.model.predict(d_X)
        pred_opt = (result > self.threshold).astype(int)
        return pred_opt

    def evaluate(self):
        # Definition de X et Y et de test
        X_cross = np.array(self.X_train)
        y_cross = np.array(self.y_train)
        id_test = self.X_test.index.values

        sub = pd.DataFrame()
        sub['id'] = id_test
        sub['target'] = np.zeros_like(id_test)
        best_ntree = 0

        for i, (train_index, test_index) in enumerate(self.skf.split(X_cross, y_cross)):
            print('[Fold %d/%d]' % (i + 1, self.kfold))
            X_train_kfd, X_valid = X_cross[train_index], X_cross[test_index]
            y_train_kfd, y_valid = y_cross[train_index], y_cross[test_index]

            # Convert our data into XGBoost format
            d_train = xgboost.DMatrix(X_train_kfd, y_train_kfd)
            d_valid = xgboost.DMatrix(X_valid, y_valid)
            d_test = xgboost.DMatrix(self.X_test.values)
            watchlist = [(d_train, 'train'), (d_valid, 'valid')]

            # Train the model! We pass in a max of 2500 rounds (with early stopping after 70)
            # and the custom metric (maximize=True tells xgb that higher metric is better)
            mdl = xgboost.train(self.params, d_train, 2500, evals=watchlist, early_stopping_rounds=60, verbose_eval=50)
            best_ntree += mdl.best_ntree_limit / self.kfold

            print('[Fold %d/%d Prediciton:]' % (i + 1, self.kfold))
            # Predict on our test data
            p_test = mdl.predict(d_test)
            sub['target'] += p_test / self.kfold

        self.num_boost_round = int(best_ntree)
        fpr, tpr, threshold = roc_curve(self.y_test, sub.target)
        self.roc_auc = auc(fpr, tpr)
        print('AUC on cross-validation :', self.roc_auc)
        print('Moyenne best tree :', best_ntree)
        return self.roc_auc

    def bayes_evaluate(self, learning_rate, max_depth, subsample, min_child_weight, gamma, colsample_bytree):
        """Compute the best parameters of XGboost

        Parameters
        --------
        learning_rate :
            Step size shrinkage used in update to prevents overfitting.

        max_depth:
            Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.

        subsample:
            Subsample ratio of the training instances.
            Setting it to 0.5 means that XGBoost would randomly
            sample half of the training data prior to growing trees.
            and this will prevent overfitting.

        min_child_weight :
            Minimum sum of instance weight (hessian) needed in a child.

        gamma :
            Minimum loss reduction required to make a further partition on a leaf node of the tree.

        colsample_bytree :
            Subsample ratio of columns when constructing each tree.

        """

        X_cross = np.array(self.X_train)
        y_cross = np.array(self.y_train)

        params = {
            'eta': learning_rate,
            'max_depth': int(max_depth),
            'subsample': max(min(subsample, 1), 0),
            'objective': 'binary:logistic',
            'base_score': np.mean(y_cross),  # base prediction = mean(target)
            'silent': 1,
            'eval_metric': 'auc',
            'min_child_weight': int(min_child_weight),
            'gamma': max(gamma, 0),
            'colsample_bytree': colsample_bytree,
        }

        id_test = self.X_test.index.values

        sub = pd.DataFrame()
        sub['id'] = id_test
        sub['target'] = np.zeros_like(id_test)

        for i, (train_index, test_index) in enumerate(self.skf.split(X_cross, y_cross)):
            X_train_kfd, X_valid = X_cross[train_index], X_cross[test_index]
            y_train_kfd, y_valid = y_cross[train_index], y_cross[test_index]

            # Convert our data into XGBoost format
            d_train = xgboost.DMatrix(X_train_kfd, y_train_kfd)
            d_valid = xgboost.DMatrix(X_valid, y_valid)
            d_test = xgboost.DMatrix(self.X_test.values)
            watchlist = [(d_train, 'train'), (d_valid, 'valid')]

            # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
            # and the custom metric (maximize=True tells xgb that higher metric is better)
            mdl = xgboost.train(params, d_train, 2500, evals=watchlist, early_stopping_rounds=60, verbose_eval=False)
            # Predict on our test data
            p_test = mdl.predict(d_test)
            sub['target'] += p_test / self.kfold

        fpr, tpr, threshold = roc_curve(self.y_test, sub.target)
        cv_result = auc(fpr, tpr)

        return cv_result

    def set_best_params(self, params):
        # Best parameters
        self.params = {
            'n_trees': 250,
            'eta': params['learning_rate'],
            'max_depth': int(params['max_depth']),
            'subsample': params['subsample'],
            'objective': 'binary:logistic',
            'base_score': np.mean(self.y_train).values[0],
            'silent': 1,
            'eval_metric': 'auc',
            'min_child_weight': int(params['min_child_weight']),
            'gamma': params['gamma'],
            'colsample_bytree': params['colsample_bytree']
        }

    def save_model(self, filename: str) -> None:
        """Save the prediction Model"""

        model = {
            "model": self.model,
            "threshold": self.threshold,
            "roc_auc": self.roc_auc,
            "num_boost_round": self.num_boost_round
        }
        super().save_model(filename, model)  # inherit of parent class Model

    def load_model(self, filename: str) -> None:
        """Load Model or intermediate model
        """

        if filename not in [config.INTERDMEDIATE_MODEL_NAME, config.MODEL_NAME]:
            raise Exception("Filename incorrect.")

        if filename == config.INTERDMEDIATE_MODEL_NAME:
            filepath = f"{config.TRAINED_INTERMEDIATE_MODEL_DIR}/{filename}{_version}.pkl"

        elif filename == config.MODEL_NAME:
            filepath = f"{config.TRAINED_MODEL_DIR}/{filename}{_version}.pkl"

        trained_model = joblib.load(filename=filepath)

        self.model = trained_model['model']
        self.threshold = trained_model['threshold']
        self.roc_auc = trained_model['roc_auc']
        self.num_boost_round = trained_model['num_boost_round']
