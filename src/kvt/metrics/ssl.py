import inspect

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import normalize


class BaseSSLScorer:
    def __init__(self, test_size=0.2, random_state=42, apply_normalize=True, **kwargs):
        self.test_size = test_size
        self.random_state = random_state
        self.model_kwargs = kwargs
        self.apply_normalize = apply_normalize

    def _get_model(self, **kwargs):
        raise NotImplementedError

    def _get_score_function(self):
        raise NotImplementedError

    def __call__(self, embeddings, target):
        if self.apply_normalize:
            embeddings = normalize(embeddings)

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        model = self._get_model(**self.model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metric_fn = self._get_score_function()
        score = metric_fn(y_test, y_pred)

        return score


class BaseSSLCVScorer:
    def __init__(
        self, n_splits=5, random_state=42, apply_normalize=True, n_jobs=5, **kwargs
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_kwargs = kwargs
        self.apply_normalize = apply_normalize
        self.n_jobs = n_jobs

    def _get_model(self, **kwargs):
        raise NotImplementedError

    def _get_score_function(self):
        raise NotImplementedError

    def __call__(self, embeddings, target):
        if self.apply_normalize:
            embeddings = normalize(embeddings)

        model = self._get_model(**self.model_kwargs)
        metric_fn = self._get_score_function()

        scores = cross_val_score(
            model,
            embeddings,
            target,
            scoring=make_scorer(metric_fn),
            cv=self.n_splits,
            n_jobs=self.n_jobs,
        )
        score = np.mean(scores)

        return score


class LogLossAfterTrainingEmbeddingsWithLogisticRegression(BaseSSLCVScorer):
    def _get_model(self, **kwargs):
        return LogisticRegression(**kwargs)

    def _get_score_function(self):
        return log_loss


class MSEAfterTrainingEmbeddingsWithLinearRegression(BaseSSLCVScorer):
    def _get_model(self, **kwargs):
        return LinearRegression(**kwargs)

    def _get_score_function(self):
        return mean_squared_error
