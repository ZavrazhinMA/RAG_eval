from sklearn.metrics import precision_score, recall_score
from tqdm.auto import tqdm
tqdm.pandas()


def calculate_precision(df, retrieved_col, relevant_col, output_col='Precision', k=None):
    """
    Рассчитывает Precision или Precision@k для DataFrame.
    
    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения Precision.
    k - (опционально) количество верхних документов для оценки Precision@k.
    
    Возвращает:
    DataFrame с добавленным столбцом Precision или Precision@k.
    """

    def _precision(row):
        retrieved_docs = row[retrieved_col]
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
        retrieved_docs = set(retrieved_docs)
        relevant_docs = set(row[relevant_col])
        relevant_found = len(retrieved_docs & relevant_docs)
        precision_value = relevant_found / len(retrieved_docs) if retrieved_docs else 0

        return precision_value

    df[output_col] = df.progress_apply(_precision, axis=1)

    return df


def precision_sklearn(df, retrieved_col, relevant_col, output_col='Precision', k=None):
    """
    Рассчитывает Precision или Precision@k для DataFrame с использованием scikit-learn.
    
    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения Precision.
    k - (опционально) количество верхних документов для оценки Precision@k.
    
    Возвращает:
    DataFrame с добавленным столбцом Precision или Precision@k.
    """

    def precision(row):
        retrieved = row[retrieved_col]
        relevant = set(row[relevant_col])
        if k is not None:
            retrieved = retrieved[:k]
        y_true = [1 if doc in relevant else 0 for doc in retrieved]
        y_pred = [1] * len(retrieved)

        if len(y_true) > 0:
            precision = precision_score(y_true, y_pred, zero_division=0)
        else:
            precision = 0.0

        return precision

    # Применяем функцию precision ко всему DataFrame
    df[output_col] = df.progress_apply(precision, axis=1)

    return df


def calculate_recall(df, retrieved_col, relevant_col, output_col='Recall', k=None):
    """
    Рассчитывает Recall или Recall@k для DataFrame.
    
    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения Recall.
    k - (опционально) количество верхних документов для оценки Recall@k.
    
    Возвращает:
    DataFrame с добавленным столбцом Recall или Recall@k.
    """

    def _recall(row):
        retrieved_docs = row[retrieved_col]
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
        retrieved_docs = set(retrieved_docs)
        relevant_docs = set(row[relevant_col])

        relevant_found = len(retrieved_docs & relevant_docs)

        recall_value = relevant_found / len(relevant_docs) if relevant_docs else 0

        return recall_value

    df[output_col] = df.progress_apply(_recall, axis=1)

    return df


def recall_sklearn(df, retrieved_col, relevant_col, output_col='Recall', k=None):
    """
    Рассчитывает Recall или Recall@k для DataFrame с использованием scikit-learn.

    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения Recall.
    k - (опционально) количество верхних документов для оценки Recall@k.

    Возвращает:
    DataFrame с добавленным столбцом Recall или Recall@k.
    """

    def _recall(row):
        retrieved = row[retrieved_col]
        relevant = set(row[relevant_col])

        if k is not None:
            retrieved = retrieved[:k]

        y_true = [1 if doc in relevant else 0 for doc in relevant]
        y_pred = [1 if doc in retrieved else 0 for doc in relevant]

        if len(relevant) > 0:
            recall = recall_score(y_true, y_pred, zero_division=0)
        else:
            recall = 0.0

        return recall

    df[output_col] = df.progress_apply(_recall, axis=1)

    return df
