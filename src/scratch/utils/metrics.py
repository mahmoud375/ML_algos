from typing import Any, Sequence, Union

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import silhouette_score as skl_silhouette_score
from sklearn.cluster import KMeans
from typing import Optional

Number = Union[int, float]
ArrayLike = Sequence[Any]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the classification accuracy.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        The fraction of correctly predicted labels.
    """
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) for regression.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    float
        The mean of the squared differences between true and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) for regression.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    float
        The square root of the mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) for regression.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    float
        The mean of the absolute differences between true and predicted values.
    """
    return np.mean(np.abs(y_true - y_pred))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Median Absolute Error for regression.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    float
        The median of the absolute differences between true and predicted values.
    """
    return np.median(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Coefficient of Determination (R²) for regression.

    R² indicates the proportion of the variance in the dependent variable that
    is predictable from the independent variable(s).

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    float
        The R² score. A value of 1 indicates perfect prediction.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Explained Variance Score for regression.

    This metric measures the proportion of the target variable's variance
    that is captured by the model.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    float
        The explained variance score.
    """
    return 1 - np.var(y_true - y_pred) / np.var(y_true)


def compute_roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) for binary classification.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels.
    y_scores : array-like, shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        The ROC AUC score.
    """
    return roc_auc_score(y_true, y_scores)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix for classification.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    array
        Confusion matrix whose i-th row and j-th column entry indicates the number of
        samples with true label being i-th class and predicted label being j-th class.
    """
    return confusion_matrix(y_true, y_pred)


def precision_score(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary", pos_label: Any = 1
) -> float:
    """
    Compute the precision score.

    Precision is defined as the ratio of true positives to the sum of true and false positives.
    It answers the question: "Of all instances predicted positive, how many are actually positive?"

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    average : {'binary', 'macro'}, default='binary'
        - 'binary': Only report results for the class specified by pos_label.
        - 'macro': Calculate metrics for each label, and find their unweighted mean.
                  This does not take label imbalance into account.
    pos_label : Any, default=1
        The class to report if average='binary'. It can be an integer or string.

    Returns
    -------
    precision : float
        Precision score.

    Example
    -------
    >>> precision_score([1, 0, 1, 1], [1, 0, 0, 1])
    0.6666666666666666
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == "binary":
        # True positives: cases where both are equal to pos_label.
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        # False positives: predicted pos_label but true label is not pos_label.
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    elif average == "macro":
        classes = np.unique(np.concatenate((y_true, y_pred)))
        prec_list = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            clazz_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            prec_list.append(clazz_prec)
        return float(np.mean(prec_list))
    else:
        raise ValueError("average must be either 'binary' or 'macro'")


def recall_score(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary", pos_label: Any = 1
) -> float:
    """
    Compute the recall score (sensitivity).

    Recall is defined as the ratio of true positives to the sum of true positives and false negatives.
    It answers the question: "Of all instances that are actually positive, how many did we predict as positive?"

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets.
    average : {'binary', 'macro'}, default='binary'
        - 'binary': Only report results for the class specified by pos_label.
        - 'macro': Calculate metrics for each label, and find their unweighted mean.
    pos_label : Any, default=1
        The class considered as positive when average == 'binary'.

    Returns
    -------
    recall : float
        Recall score.

    Example
    -------
    >>> recall_score([1, 0, 1, 1], [1, 0, 0, 1])
    0.6666666666666666
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    elif average == "macro":
        classes = np.unique(np.concatenate((y_true, y_pred)))
        recall_list = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            clazz_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_list.append(clazz_recall)
        return float(np.mean(recall_list))
    else:
        raise ValueError("average must be either 'binary' or 'macro'")


def f1_score(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary", pos_label: Any = 1
) -> float:
    """
    Compute the F1 score, which is the harmonic mean of precision and recall.

    The F1 score balances the need for high precision and high recall:
        F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets.
    average : {'binary', 'macro'}, default='binary'
        - 'binary': Report F1 score for the class specified by pos_label.
        - 'macro': Calculate F1 for each label, and return the unweighted mean.
    pos_label : Any, default=1
        The class considered as positive when average == 'binary'.

    Returns
    -------
    f1 : float
        F1 score.

    Example
    -------
    >>> f1_score([1, 0, 1, 1], [1, 0, 0, 1])
    0.6666666666666666
    """
    prec = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def silhouette_score(
    X: np.ndarray, labels: Optional[np.ndarray] = None, n_clusters: Optional[int] = None, random_state: Optional[int] = None
) -> float:
    """
    Compute the Silhouette Score for clustering evaluation.

    The Silhouette Score measures how similar an object is to its own cluster 
    compared to other clusters. A higher silhouette score indicates better-defined clusters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature data used for clustering.
    labels : array-like of shape (n_samples,), optional
        Predicted cluster labels. If None, KMeans clustering will be applied automatically.
    n_clusters : int, optional
        Number of clusters to use if labels are not provided.
    random_state : int, optional
        Random seed for KMeans initialization (for reproducibility).

    Returns
    -------
    float
        Silhouette Score ranging from -1 to 1.

    Notes
    -----
    - Score close to +1: samples are well clustered.
    - Score close to 0: clusters are overlapping.
    - Score close to -1: samples are misassigned to the wrong clusters.

    Example
    -------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    >>> silhouette_score(X, n_clusters=3)
    0.6554
    """
    if labels is None:
        if n_clusters is None:
            raise ValueError("You must provide either 'labels' or 'n_clusters'.")
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(X)
    
    return skl_silhouette_score(X, labels)