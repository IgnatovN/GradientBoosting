"""Implementation of Gradient Boosting"""

import numpy as np

from decision_tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    """Gradient boosting regressor."""

    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            loss="mse",
            verbose=False,
            subsample_size=0.5,
            replace=False
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.trees_ = []
        self.base_pred_ = None
        self.subsample_size = subsample_size
        self.replace = replace

    def _mse(self, y_true, y_pred):
        """Mean squared error loss function and gradient."""
        loss = np.mean((y_pred - y_true) ** 2)
        grad = (y_pred - y_true) ** 2
        return loss, grad

    def _mae(self, y_true, y_pred):
        """Mean absolute error loss function and gradient."""
        loss = np.mean(abs(y_pred - y_true))
        grad = np.sign(y_true - y_pred)

        return loss, grad

    def _subsample(self, X, y):
        """Select random subsample"""
        subsample_size = int(X.shape[0] * self.subsample_size)
        idx = np.array(X.shape[0])
        subsample_idx = np.random.choice(idx, subsample_size, replace=self.replace)
        sub_X, sub_y = X[subsample_idx], y[subsample_idx]

        return sub_X, sub_y

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        X, y = X.to_numpy(), y.to_numpy()
        self.base_pred_ = np.mean(y)
        preds = self.base_pred_.copy()

        for i in range(self.n_estimators):

            if i == 0:
                antigrad = y
            else:
                antigrad = y - preds

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            sub_X, sub_y = GradientBoostingRegressor._subsample(self, X, antigrad)

            tree.fit(sub_X, sub_y)

            b = tree.predict(X).reshape([X.shape[0]])
            self.trees_.append(tree)
            preds += self.learning_rate * b

            if self.loss == 'mse':
                mse, grad = GradientBoostingRegressor._mse(self, y, preds)

            elif self.loss == 'mse':
                mse, grad = GradientBoostingRegressor._mse(self, y, preds)
            else:
                mse = self.loss(y, preds)

            if self.verbose:
                print(mse)

        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        X = X.to_numpy()
        predictions = self.base_pred_

        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X).reshape([X.shape[0]])

        return predictions
