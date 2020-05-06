"""
Constant used by the project are set here for better maintainability.
Import this module into a project's file to use any constant in it.
"""

import pathlib
import src
import pandas as pd

pd.options.display.max_rows = 100
pd.options.display.max_columns = 10

# Root dir
# ================================================
PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
TRAINED_PIPELINE_DIR = PACKAGE_ROOT / "trained_models/processed_pipeline"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models/processed_model"
TRAINED_MODEL_METRICS_DIR = PACKAGE_ROOT / "trained_models/metrics"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# Data
# ================================================
EXCEL_FILE = "credit.xlsx"
DATA_FILE = 'credit.csv'
TRAINING_PROCESSED = 'processed_data.csv'
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"

TARGET = ['default']

# variables
FEATURES = [
    "checking_balance",
    "months_loan_duration",
    "credit_history",
    "purpose",
    "amount",
    "savings_balance",
    "employment_length",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_history",
    "property",
    "age",
    "installment_plan",
    "housing",
    "existing_credits",
    "dependents",
    "telephone",
    "foreign_worker",
    "job",
]

MAPPING_TARGET = {1: 0, 2: 1}  # remap 1 -> 0 (good loan), 2 -> 1 (defaulting)



# SAVE PIPELINE CONFIG
# ================================================
PIPELINE_PREPROCESSING_FEATURES_NAME = "preprocessing_features_pipeline"
PIPELINE_PREPROCESSING_TARGET_NAME = "preprocessing_target_pipeline"



MODEL_NAME = "lightgbm_output_v"

METRICS_FILE = "report.csv"


# # used for differential testing
# ACCEPTABLE_MODEL_DIFFERENCE = 0.05
