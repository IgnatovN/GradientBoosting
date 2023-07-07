from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.array, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        mean = np.mean(y)
        return np.mean(np.square(y - mean))

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        return (DecisionTreeRegressor._mse(self, y_left) * len(y_left) + DecisionTreeRegressor._mse(
            self, y_right) * len(y_right)) / (len(y_left) + len(y_right))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_thr = None
        best_idx = None
        best_mse = float("inf")

        for idx in range(X.shape[1]):
            for thr in np.unique(X[:, idx]):
                current_mse = DecisionTreeRegressor._weighted_mse(self,
                                                                  y[np.where(X[:, idx] <= thr)[0]],
                                                                  y[np.where(X[:, idx] > thr)[0]])
                if current_mse < best_mse:
                    best_thr = thr
                    best_mse = current_mse
                    best_idx = idx

        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        node = Node(
            value=round(np.mean(y)),
            n_samples=X.shape[0],
            mse=DecisionTreeRegressor._mse(self, y),
        )

        if depth < self.max_depth and X.shape[0] >= self.min_samples_split:
            node.feature, node.threshold = DecisionTreeRegressor._best_split(self, X, y)

            X_left = X[np.where(X[:, node.feature] <= node.threshold)[0]]
            y_left = y[np.where(X[:, node.feature] <= node.threshold)[0]]
            X_right = X[np.where(X[:, node.feature] > node.threshold)[0]]
            y_right = y[np.where(X[:, node.feature] > node.threshold)[0]]

            node.left = DecisionTreeRegressor._split_node(
                self,
                X_left,
                y_left,
                depth=depth + 1
            )
            node.right = DecisionTreeRegressor._split_node(
                self,
                X_right,
                y_right,
                depth=depth + 1
            )

        return node

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        return DecisionTreeRegressor._as_json(self, self.tree_)

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        json = '{'

        for attr, value in node.__dict__.items():
            if value:
                if type(value) == Node:
                    json += f'"{attr}": {DecisionTreeRegressor._as_json(self, value)}, '

                elif attr == 'mse':
                    json += f'"{attr}": {round(value, 2)}, '

                elif attr == 'value' and type(node.__dict__['left']) == Node:
                    continue

                else:
                    json += f'"{attr}": {value}, '

        json = json[:-2] + '}'

        return json

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        predicts = []

        for sample in X:
            predicts.append(DecisionTreeRegressor._predict_one_sample(self, sample))

        return np.array(predicts)

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        prediction = None
        node = self.tree_

        while prediction is None:
            if node.left is not None:

                if features[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            else:
                prediction = node.value

        return prediction
