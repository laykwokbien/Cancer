import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model import KNeighborsClassfier

def find_best_k(X_train, y_train, test_data, y_true, k_value):
    if not isinstance(k_value, np.ndarray):
        raise ValueError('K Value not in numpy.ndarray')
    
    total_accuracy = []
    for k in k_value:
        knn = KNeighborsClassfier(n_neighbors=k, p=3, method='auto', leaf_node=30, metrics='euclidean')
        knn.fit(X_train, y_train)
        pred = knn.predict(test_data)
        accuracy = accuracy_error(y_true, pred)[0]
        total_accuracy.append(int(accuracy * 100))

    plt.title('Cross Validation')    
    plt.plot(k_value, total_accuracy, marker='o')
    plt.show()

def accuracy_error(y_true, y_pred):
    accuracy = sum(1 for y_i, y_hat in zip(y_true, y_pred) if y_i == y_hat) / len(y_true)
    error = accuracy - 1
    return accuracy, error

def confusion_matrix(y_true, y_pred, decode, cmap='magma', fmt='d'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(y_true)
    n_labels = len(labels)
    
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    for y_i, y_hat in zip(y_true, y_pred):
        matrix[y_hat, y_i] += 1
    
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap,
                yticklabels=[f'Predicted {decode[label]}' for label in labels],
                xticklabels=[f'Actual {decode[label]}' for label in labels])
    plt.title('Confusion Matrix')
    plt.show()
    
def one_hot_encode(y, n_label):
    return np.eye(len(y), n_label)[y]

def f1score(y_true, y_pred, average='micro'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(y_true)
    n_label = len(labels)

    y_true = one_hot_encode(y_true, n_label)
    y_pred = one_hot_encode(y_pred, n_label)

    totaltp, totalfp, totalfn = 0, 0, 0

    f1_score = []
    precision_score = []
    recall_score = []

    weights = np.sum(y_true, axis=0)

    for i in range(n_label):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        fn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)


        f1 = 2 * (precision * recall) / (precision + recall)

        f1_score.append(f1)
        precision_score.append(precision)
        recall_score.append(recall)

        totaltp += tp
        totalfp += fp
        totalfn += fn
    
    if average == 'macro':
        precision_score = np.mean(precision_score)
        recall_score = np.mean(recall_score)
        f1_score = np.mean(f1_score)
    elif average == 'micro':
        precision_score = totaltp / (totaltp + totalfp)
        recall_score = totaltp / (totaltp + totalfn)
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    elif average == 'weighted':
        total_weight = np.sum(weights)
        precision_score = np.sum((np.array(precision_score) * weights) / total_weight) 
        recall_score = np.sum((np.array(recall_score) * weights) / total_weight)
        f1_score = np.sum((np.array(f1_score) * weights) / total_weight)

    return f1_score, precision_score, recall_score