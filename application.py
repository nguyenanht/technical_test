from api.app import create_app
from api.config import DevelopmentConfig, ProductionConfig

from flask_cors import CORS

application = create_app(
    config_object=ProductionConfig)  # Can be DevelopmentConfig or ProductionConfig
CORS(application)

if __name__ == '__main__':
    application.run(host='0.0.0.0')
