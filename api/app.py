from flask import Flask

from api.config import get_logger

_logger = get_logger(logger_name=__name__)


def create_app(*, config_object) -> Flask:
    """Create a flask api instance."""

    flask_app = Flask('flask')
    flask_app.config.from_object(config_object)

    # import blueprints
    from api.controller import application
    flask_app.register_blueprint(application)
    _logger.debug('Application instance created')

    return flask_app
