import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import ast


def weighted_f1(y_true, y_pred, weights):
    """Расчет взвешенной F1 меры с индивидуальными весами для классов."""

    # преобразуем метки из строкового представления списка в список
    y_true_converted = [ast.literal_eval(label) if isinstance(label, str) else label for label in y_true]
    y_pred_converted = [ast.literal_eval(label) if isinstance(label, str) else label for label in y_pred]

    y_true_flat = [label for sublist in y_true_converted for label in sublist]
    y_pred_flat = [label for sublist in y_pred_converted for label in sublist]

    # рассчитываем F1 для всех классов
    _, _, f1, support = precision_recall_fscore_support(y_true_flat, y_pred_flat, average=None,
                                                                     labels=list(weights.keys()))

    # вычисление взвешенной F-меры
    weighted_f1 = np.sum(f1 * [weights[label] for label in np.unique(y_true_flat + y_pred_flat)] * support) / np.sum(
        support * [weights[label] for label in np.unique(y_true_flat + y_pred_flat)])
    return weighted_f1


def evaluate(ref_uri, pred_uri, class_weights):
    # чтение данных
    ref_df = pd.read_csv(ref_uri)
    pred_df = pd.read_csv(pred_uri)

    # получение меток
    y_true = ref_df['label'].tolist()
    y_pred = pred_df['label'].tolist()

    weighted_f1_score = weighted_f1(y_true, y_pred, class_weights)

    return {
        'Weighted F1 Score': weighted_f1_score,
    }


# if __name__ == "__main__":
ref_uri = 'privat_test.csv'
pred_uri = 'gt_test.csv'
class_weights = {'B-discount': 1, 'B-value': 2, 'I-value': 2, 'O': 0.003}

metrics = evaluate(pred_uri, ref_uri, class_weights)

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")
