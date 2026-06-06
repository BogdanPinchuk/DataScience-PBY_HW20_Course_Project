import sqlite3
import kagglehub
import numpy as np
import pandas as pd
import apps.reporter as rpt
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.core.display import Markdown
from pandas import Series, DataFrame
from pandas.io.formats.style import Styler
from kagglehub import KaggleDatasetAdapter
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, classification_report,
                             roc_auc_score, average_precision_score, mean_absolute_percentage_error,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score, precision_recall_curve, roc_curve)


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


def calc_class_metrics(y_test, y_pred, y_prob=None, charts=True, figsize: tuple[float, float] = (10, 4)) -> Styler:
    """
    Calc and print a classifier metrics
    :param y_test: original target test set
    :param y_pred: predicted set
    :param y_prob: probabilities set
    :param charts: show charts
    :param figsize: figure size
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

    # Graphic results
    if y_prob is not None and charts:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        _, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], '--')
        ax.fill_between(fpr, tpr, alpha=0.25)
        ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
        # ax.minorticks_on()
        # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
        ax.set_title("ROC Curve", pad=10, loc='center', color='black')
        ax.set_xlabel("FPR", labelpad=10, loc='center', color='black')
        ax.set_ylabel("RPT", labelpad=10, loc='center', color='black')

        ax = axes[1]
        ax.plot(rec, prec)
        ax.plot([1, 0], [0, 1], '--')
        ax.fill_between(rec, prec, alpha=0.25)
        ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
        # ax.minorticks_on()
        # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
        ax.set_title("Precision-Recall Curve", pad=10, loc='center', color='black')
        ax.set_xlabel("Recall", labelpad=10, loc='center', color='black')
        ax.set_ylabel("Precision", labelpad=10, loc='center', color='black')

        plt.suptitle(f"Model Performance Curves")

        plt.tight_layout()
        plt.show()

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


def calc_outliers(ids: Series, data: Series) -> tuple:
    """
    Calculation and finding outlier
    :param ids: IDs
    :param data: input data
    :return: list of tuples (feature name, number of outliers, ID list of outliers)
    """
    feature_name = data.name
    # Calculating outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    merged_data = pd.concat([ids, data], axis=1)
    outliers = [int(idx) for (idx, value) in merged_data.values if value < lower_limit or value > upper_limit]
    return feature_name, len(outliers), outliers


def analys_feature_coef(scores: Series, figsize: tuple[float, float] = (12, 5)) -> None:
    """
    Display a table of feature coefficients and draw charts,
    :param scores: series of scores
    :param figsize: figure size
    """
    features = scores.index
    scores_abs = np.abs(scores)
    feature_importances = scores / scores_abs.sum()
    feature_importances_abs = scores_abs / scores_abs.sum()
    feature_importances_cum = np.cumsum(feature_importances_abs)

    importances_df = (
        pd.DataFrame({"feature": features, "coef": feature_importances, "coef_abs": feature_importances_abs}).
        sort_values(by="coef_abs", ascending=False, ignore_index=True))

    # Print results
    display(importances_df.style.set_caption("Feature Importances"))

    # Graphic results
    x_axis = np.arange(len(feature_importances_cum))

    _, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.barh(importances_df["feature"], importances_df["coef"])
    ax.invert_yaxis()
    ax.grid(axis='x', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
    # ax.minorticks_on()
    # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
    ax.set_title("Feature Coefficients", pad=10, loc='center', color='black')
    ax.set_xlabel("Coefficient", labelpad=10, loc='center', color='black')
    ax.set_ylabel("Feature", labelpad=10, loc='center', color='black')

    ax = axes[1]
    ax.plot(x_axis, feature_importances_cum)
    ax.fill_between(x_axis, feature_importances_cum, alpha=0.25)
    ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
    # ax.minorticks_on()
    # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
    ax.set_title("Cumulative Feature Coefficients", pad=10, loc='center', color='black')
    ax.set_xlabel("Number of Features", labelpad=10, loc='center', color='black')
    ax.set_ylabel("Cumulative Coefficient", labelpad=10, loc='center', color='black')

    plt.suptitle(f"Analysis of the Feature Coefficients")

    plt.tight_layout()
    plt.show()
