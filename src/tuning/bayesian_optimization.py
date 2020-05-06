from bayes_opt import BayesianOptimization


class BayesOpt:

    def __init__(self, model_to_tune, evaluate, params):
        """
        params
        ---------
        evaluate: function
            function for optimization
        params: dict
            dict of params where the BayesianOptimization find the best

        """
        self.model = BayesianOptimization(evaluate, params)
        self.best_params = None
        self.model_to_tune = model_to_tune

    def tune(self, init_points, n_iter) -> dict:
        """ Find the best hyperparameters for the model

        """
        # Compute optimal parameters
        self.model.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
        self.best_params = self.model.max['params']
        return self.best_params








