import sqlite3
import kagglehub
import numpy as np
import pandas as pd
import apps.reporter as rpt
from pandas import Series, DataFrame
from pandas.io.formats.style import Styler
from kagglehub import KaggleDatasetAdapter
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, classification_report,
                             roc_auc_score, average_precision_score, mean_absolute_percentage_error,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score)


def download_and_extract_from_kagglehub(ds_path: str,
                                        ds_file_name: str,
                                        db_file_name: str,
                                        ds_name: str = None,
                                        update_db: bool = False) -> DataFrame | None:
    """
    Download and extract data from kagglehub
    :param ds_path: path to kaggle dataset
    :param ds_file_name: name of kaggle dataset
    :param db_file_name: name of a database file
    :param ds_name: dataset name
    :param update_db: update the database
    :return: DataFrame or None
    """
    if ds_name is None:
        ds_name = ds_path.split("/")[-1].replace('-', '_')

    # Note: to handle error: "SSL: CERTIFICATE_VERIFY_FAILED" or no connection to the server
    try:
        # for testing
        # raise Exception

        ds_data = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            ds_path,
            ds_file_name,
        )

        # Use only one time to initialize/update data (at first time)
        if not ds_data.empty:
            conn = sqlite3.connect(db_file_name)
            try:
                exist_conf = "replace" if update_db else "fail"
                ds_data.to_sql(ds_name, conn, if_exists=exist_conf, index=False)
            except ValueError:
                pass
            finally:
                conn.close()
    except Exception:
        conn = sqlite3.connect(db_file_name)
        ds_data = pd.read_sql(f"SELECT * FROM {ds_name}", conn)
        conn.close()

    return ds_data


def train_test_split_by_order(array, test_size: float) -> tuple:
    """
    Split arrays or matrices into order to train and test subset
    :param array: input data
    :param test_size: the proportion of the dataset for the test split
    :return: tuple of train and test arrays
    """
    n_samples = len(array)
    if n_samples <= 1:
        raise ValueError("The array must contain more than one element!")

    if test_size <= 0.0 or 1.0 <= test_size:
        raise ValueError("Invalid test size, it should be between 0.0 and 1.0!")

    n_test = int(n_samples * test_size)
    if n_test == 0:
        n_test = 1

    if isinstance(array, (Series, DataFrame)):
        train_array = array.iloc[:-n_test]
        test_array = array.iloc[-n_test:]
    else:
        train_array = array[:-n_test]
        test_array = array[-n_test:]

    return train_array, test_array


def calc_class_metrics(y_test, y_pred, y_prob=None) -> Styler:
    """
    Calc and print a classifier metrics
    :param y_test: original target test set
    :param y_pred: predicted set
    :param y_prob: probabilities set
    :return: DataFrame styler
    """
    rp = rpt.Reporter()
    rp.tolerance = 4

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    rp.add_item("Confusion Matrix", rp.format_matrix(cm))
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    rp.add_item("Accuracy\n(Точність)", rp.format_value(accuracy))
    # Precision
    precision = precision_score(y_test, y_pred)
    rp.add_item("Precision\n(Влучність)", rp.format_value(precision))
    # Recall
    recall = recall_score(y_test, y_pred)
    rp.add_item("Recall\n(Повнота)", rp.format_value(recall))
    # F1-score
    f1 = f1_score(y_test, y_pred)
    rp.add_item("F1-score", rp.format_value(f1))

    if y_prob is not None:
        # ROC-AUC
        auc = roc_auc_score(y_test, y_prob)
        rp.add_item("Receiver Operating Characteristic\n(ROC-AUC)", rp.format_value(auc))
        # AP
        ap = average_precision_score(y_test, y_prob)
        rp.add_item("Average Precision (AP)", rp.format_value(ap))

    df = rp.get_pd_report()

    # Print results
    rp.print_pd_report(f"Метрики класифікації")
    print(classification_report(y_test, y_pred))

    return df


def calc_regres_metrics(y_test, y_pred) -> Styler:
    """
    Calc and print a regressor metrics
    :param y_test: original target test set
    :param y_pred: predicted set
    :return: DataFrame styler
    """
    rp = rpt.Reporter()
    rp.tolerance = 4

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    rp.add_item("MAE", rp.format_value(mae))  # type: ignore
    # Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    rp.add_item("MSE", rp.format_value(mse))  # type: ignore
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    rp.add_item("RMSE", rp.format_value(rmse))  # type: ignore
    # R2 - coefficient of determination
    r2 = r2_score(y_test, y_pred)
    rp.add_item(f"R²\n(коефіцієнт детермінації)", rp.format_value(r2))
    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rp.add_item("MAPE", rp.format_value(mape))

    df = rp.get_pd_report()

    # Print results
    rp.print_pd_report(f"Метрики регресії")

    return df


def calc_cluster_metrics(data_set: DataFrame, labels_pred, labels_true=None) -> Styler:
    """
    Calc and print the internal and external cluster metrics
    :param data_set: input dataset
    :param labels_pred: predicted labels
    :param labels_true: true labels
    :return: DataFrame styler
    """
    unique_clusters = len(set(labels_pred))

    rp = rpt.Reporter()
    rp.tolerance = 4
    rp.add_item("Кількість кластерів", str(unique_clusters))

    # Internal metrics
    if unique_clusters > 1:
        # Silhouette Score
        sil = silhouette_score(data_set.values, labels_pred)
        rp.add_item("Silhouette Score", rp.format_value(sil))
        # Calinski-Harabasz
        ch = calinski_harabasz_score(data_set.values, labels_pred)
        rp.add_item("Calinski-Harabasz Index", rp.format_value(ch))
        # Davies-Bouldin
        db = davies_bouldin_score(data_set.values, labels_pred)
        rp.add_item("Davies-Bouldin Index", rp.format_value(db))

        if labels_true is not None:
            # Adjusted Rand
            ari = adjusted_rand_score(labels_true, labels_pred)
            rp.add_item("Adjusted Rand Index (ARI)", rp.format_value(ari))
            # Normalized Mutual Information
            nmi = normalized_mutual_info_score(labels_true, labels_pred)
            rp.add_item("Normalized Mutual Information (NMI)", rp.format_value(nmi))

    df = rp.get_pd_report()

    # Print results
    if labels_true is None:
        rp.print_pd_report(f"Внутрішні метрики")
    else:
        rp.print_pd_report(f"Внутрішні й зовнішні метрики")
    return df


def contrast(val1: np.number, val2: np.number) -> float:
    """
    Contrast between 2 values
    :param val1: first value
    :param val2: second value
    :return: result between 0 and 1
    """
    return abs(val1 - val2) / abs(val1 + val2) if (val1 + val2) != 0 else 0
