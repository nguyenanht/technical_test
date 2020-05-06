import pandas as pd
from flask import Blueprint, request, jsonify
from src.extraction.data_management import Data
from src.predict import make_prediction, get_models_list
from src import __version__ as _version

from api.config import get_logger
from api.validation import validate_inputs
from api import __version__ as api_version


_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/', methods=['GET'])
def home():
    return "<h1 style='color:blue'>Hello There :) !</h1>"


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@prediction_app.route('/predict/', methods=['POST'])
def predict():
    """Output of default model based on last timestamp"""
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        # _logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using marshmallow schema
        input_data, errors = validate_inputs(input_data=json_data)

        # Step 3: Model prediction
        result = make_prediction(input_data=input_data)
        _logger.debug(f'Outputs: {result}')

        # Step 4: Convert numpy ndarray to list
        predictions = result.get('predictions').tolist()
        version = result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})


@prediction_app.route('/models/', methods=['GET'])
def get_models():
    models_list = get_models_list()

    return jsonify({'models': models_list})


@prediction_app.route('/output/<model_id>', methods=['POST'])
def output_specific_model(model_id):
    model_id = model_id
    # lightgbm_output_v0.1-1588759220.335498

    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        data = request.get_json()

        # Step 2: Validate the input using marshmallow schema
        data, errors = validate_inputs(input_data=data)

        # Step 3: Model prediction
        result = make_prediction(input_data=data, id_model=model_id)

        predictions = result.get('predictions').tolist()
        version = result.get('version')

    return jsonify({
        'result': predictions,
        'model': model_id,
        'version': version,
        'errors': errors
    })


@prediction_app.route('/outputs/<model_id>', methods=['POST'])
def batch_output_specific_model(model_id):
    """Output of a specific model with test.csv data

    parameters
    ---------------
        model_id: str

    returns
    ---------------

    """
    model_id = model_id
    # lightgbm_output_v0.1-1588759220.335498

    if request.method == 'POST':
        data_mngmnt = Data()
        data_mngmnt.from_csv("test.csv", sep=',')
        data = data_mngmnt.df
        result = make_prediction(input_data=data, id_model=model_id)
        predictions = result.get('predictions').tolist()

    return jsonify({
        'result': predictions,
        'model': model_id
    })


@prediction_app.route('/outputs_upload/<model_id>', methods=['POST'])
def outputs_upload(model_id):
    if request.method == 'POST':
        # Create variable for uploaded file
        df = pd.read_csv(request.files.get('fileupload'))

        result = make_prediction(input_data=df, id_model=model_id)
        predictions = result.get('predictions').tolist()

    return jsonify({
        'result': predictions,
        'model': model_id
    })
