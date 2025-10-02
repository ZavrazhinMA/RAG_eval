from sklearn.metrics import precision_score, recall_score, fbeta_score, ndcg_score, average_precision_score
from tqdm.auto import tqdm
import numpy as np

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


def fbeta_sklearn(df, retrieved_col, relevant_col, output_col='F-beta', k=None, beta=1):
    """
    Рассчитывает F-beta или F1-score для DataFrame с использованием scikit-learn.
    
    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения F-beta.
    k - (опционально) количество верхних документов для оценки F-beta@k.
    beta - параметр, определяющий вес Recall относительно Precision (по умолчанию 1 для F1-score).
    
    Возвращает:
    DataFrame с добавленным столбцом F-beta или F1-score.
    """

    def _fbeta(row):
        retrieved = row[retrieved_col]
        relevant = set(row[relevant_col])

        if k is not None:
            retrieved = retrieved[:k]

        y_true = [1 if doc in relevant else 0 for doc in relevant]
        y_pred = [1 if doc in retrieved else 0 for doc in relevant]

        if len(relevant) > 0:
            fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        else:
            fbeta = 0.0

        return fbeta

    df[output_col] = df.progress_apply(_fbeta, axis=1)

    return df


def calculate_mrr(df, retrieved_col, relevant_col, output_col='RR', verbose=True, return_mrr=True):
    """
    Рассчитывает MRR для DataFrame.

    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения MRR.

    Возвращает:
    DataFrame с добавленным столбцом MRR.
    """

    def _reciprocal_rank(row):
        retrieved_docs = row[retrieved_col]
        relevant_docs = set(row[relevant_col])

        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in relevant_docs:
                return 1 / rank

        return 0

    df[output_col] = df.progress_apply(_reciprocal_rank, axis=1)
    mrr = round(df[output_col].mean(), 3)
    if verbose:
        print(f'MRR = {mrr}')
    if return_mrr:
        return df, mrr
    else:
        return df


def calculate_map(df, retrieved_col, relevant_col, output_col='AP', k=None, verbose=True, return_map=True):
    """
    Рассчитывает MAP для DataFrame.

    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения AP.
    k - (опционально) количество верхних документов для оценки AP@K (по умолчанию None, что означает использовать все документы).
    verbose - выводит значение MAP, если True.
    return_map - возвращает значение MAP, если True.

    Возвращает:
    DataFrame с добавленным столбцом AP и, опционально, значение MAP.
    """

    def _average_precision(row):
        retrieved_docs = row[retrieved_col]
        relevant_docs = set(row[relevant_col])
        if k is not None:
            retrieved_docs = retrieved_docs[:k]

        y_true = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
        y_score = np.arange(len(retrieved_docs), 0, -1)
        relevant_found = 0
        ap_sum = 0.0
        total_relevant = sum(y_true)
        if total_relevant == 0:
            return 0.0

        for i, (true_label, score) in enumerate(zip(y_true, y_score), start=1):
            if true_label == 1:
                relevant_found += 1
                precision_at_i = relevant_found / i
                ap_sum += precision_at_i

        return ap_sum / total_relevant

    df[output_col] = df.progress_apply(_average_precision, axis=1)
    map_value = round(df[output_col].mean(), 3)

    if verbose:
        print(f'MAP@{k} = {map_value}')

    if return_map:
        return df, map_value
    else:
        return df


def map_sklearn(df, retrieved_col, relevant_col, output_col='AP_sklearn', k=None, verbose=True, return_map=True):
    """
    Рассчитывает MAP для DataFrame с использованием sklearn.

    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения AP.
    k - (опционально) количество верхних документов для оценки AP@K (по умолчанию None, что означает использовать все документы).
    verbose - выводит значение MAP, если True.
    return_map - возвращает значение MAP, если True.

    Возвращает:
    DataFrame с добавленным столбцом AP и, опционально, значение MAP.
    """

    def _sklearn_average_precision(row):
        retrieved_docs = row[retrieved_col]
        relevant_docs = set(row[relevant_col])

        if k is not None:
            retrieved_docs = retrieved_docs[:k]

        y_true = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
        y_score = np.arange(len(retrieved_docs), 0, -1)

        if sum(y_true) == 0:
            return 0.0  # Если нет релевантных документов, AP = 0

        return average_precision_score(y_true, y_score)

    df[output_col] = df.progress_apply(_sklearn_average_precision, axis=1)
    map_value = round(df[output_col].mean(), 3)

    if verbose:
        print(f'MAP@{k} (sklearn) = {map_value}')

    if return_map:
        return df, map_value
    else:
        return df


def calculate_ndcg_at_k(df, retrieved_col, relevant_col, output_col='NDCG@K', k=5, verbose=True, return_ndcg=True):
    """
    Рассчитывает NDCG@K для DataFrame.

    Параметры:
    df - DataFrame с данными.
    retrieved_col - название столбца с найденными документами.
    relevant_col - название столбца с релевантными документами.
    output_col - название выходного столбца для сохранения NDCG@K.
    k - количество верхних документов для оценки NDCG@K.
    verbose - выводит значение NDCG@K, если True.
    return_ndcg - возвращает значение NDCG@K, если True.

    Возвращает:
    DataFrame с добавленным столбцом NDCG@K и, опционально, значение среднего NDCG@K.
    """

    def _dcg_at_k(retrieved_docs, relevant_docs, k):
        """
        Рассчитывает DCG@K.
        """
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], start=1):
            if doc in relevant_docs:
                dcg += 1 / np.log2(i + 1)  # Используем логарифм с основанием 2 для скидки
        return dcg

    def _idcg_at_k(relevant_docs, k):
        """
        Рассчитывает идеальный DCG@K (IDCG@K).
        """
        idcg = 0.0
        for i in range(min(len(relevant_docs), k)):
            idcg += 1 / np.log2(i + 2)
        return idcg

    def _ndcg_at_k(row):
        retrieved_docs = row[retrieved_col]
        relevant_docs = set(row[relevant_col])

        dcg_k = _dcg_at_k(retrieved_docs, relevant_docs, k)
        idcg_k = _idcg_at_k(relevant_docs, k)

        return dcg_k / idcg_k if idcg_k > 0 else 0

    df[output_col] = df.progress_apply(_ndcg_at_k, axis=1)
    avg_ndcg = round(df[output_col].mean(), 3)

    if verbose:
        print(f'MEAN NDCG@{k} = {avg_ndcg}')

    if return_ndcg:
        return df, avg_ndcg
    else:
        return df

