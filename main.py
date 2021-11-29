import pandas as pd
import time
from Metrics import *
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from CrossValidation import *
from DecisionTree import DecisionTree

INPATH = 'data/'
N_CLASSES = 2


def write_out_tree(out, filename='data/out.txt'):
    f = open(filename, "w")
    f.write(out)
    f.close()


def get_continuous_attrs(dataset):
    index = []
    for i, column in enumerate(dataset.columns):
        if dataset[column].dtype == np.float64 or dataset[column].dtype == np.int64:
            index.append(i)
    return index


def threshold_data(data, thresholds, index):
    for i, ind in enumerate(index):
        data[:, ind][data[:, ind] <= thresholds[i]] = 0
        data[:, ind][data[:, ind] > thresholds[i]] = 1
    return data


def two_way_partition(data, index):
    t = time.time()
    thresholds = []
    discrete_column = np.zeros(data.shape[0])

    for i in index:
        column = np.sort(np.unique(np.copy(data[:, i])))
        best_threshold = 0
        best_entropy = 0
        values = np.unique(column)
        thr = (values[:-1] + values[1:]) / 2
        #dc = np.zeros(data.shape[0])
        #dc[data[:, i] > thr] = 1
        #entropy_val = entropy(dc, np.unique(discrete_column))

        for threshold in thr:
            discrete_column[:] = 0
            discrete_column[data[:, i] > threshold] = 1
            entropy_val = entropy(discrete_column, np.unique(discrete_column))
            if entropy_val > best_entropy:
                best_entropy = entropy_val
                best_threshold = threshold
        data[:, i] = 0
        data[:, i][data[:, i] > best_threshold] = 1
        thresholds.append(best_threshold)
    print("two_way_partition time: ", time.time() - t)
    return data, thresholds


def continuous_to_discrete_attr(data, index, get_bins=True, bins=None, n=2):
    bins_list = []
    for ind, i in enumerate(index):
        if get_bins:
            intervals, bins = pd.qcut(data[:, i], n, retbins=get_bins, duplicates='drop')
            data[:, i] = pd.cut(data[:, i], bins=bins, labels=False, include_lowest=True)
            bins_list.append(bins)
        else:
            data[:, i] = pd.cut(data[:, i], bins=bins[ind], labels=False, include_lowest=True)

    if get_bins:
        return data, bins_list
    else:
        return data


def standard_norm(data, index):
    st = StandardScaler()
    data[:, index] = st.fit_transform(data[:, index])
    return data, st


def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',', skipinitialspace=True)
    return dataset


def main():

    dataset = load_dataset(INPATH+'adult.data')
    """
    dataset = dataset.replace({'?': np.NaN})
    dataset = dataset.dropna()
    """
    data = dataset.to_numpy()
    np.random.shuffle(data)
    continuous_attr_index = get_continuous_attrs(dataset)
    #data, bins_list = continuous_to_discrete_attr(data, continuous_attr_index, n=N_CLASSES)
    data, thresholds = two_way_partition(data, continuous_attr_index)
    X = data[:, :-1]
    Y = data[:, -1]

    print(X.shape, data.shape)
    decision_tree = DecisionTree(criterion="gini")
    decision_tree.fit(X, Y, dataset.columns[:-1])
    write_out_tree(str(decision_tree))

    # ------------------------------------------- TEST WITH TRAINING SET --------------------------------------------- #
    print("\n\n--------------------Test with Training set--------------------\n")
    for i in range(min(dataset.shape[0], 20)):
        x = X[i, :]
        ground_truth = Y[i]
        predict = decision_tree.predict(x)
        print("Index ["+str(i)+"]; Prediction: "+str(predict)+" GT: "+str(ground_truth)+"     "+("Correct" if ground_truth==predict else "Incorrect"))

    # ------------------------------------------------ CROSS VAL ----------------------------------------------------- #
    t = time.time()
    print("\n\nCross Validation Score: ", cross_val_score(decision_tree, X, Y, scoring="accuracy"))
    print("CrossValidation elapsed time: ", time.time() - t)
    # ------------------------------------------------ TEST DATA ----------------------------------------------------- #
    test_dataset = load_dataset(INPATH + 'adult.test')
    """
    test_dataset = test_dataset.replace({'?': np.NaN})
    test_dataset = test_dataset.dropna()
    """
    test_data = test_dataset.to_numpy()
    #test_data = continuous_to_discrete_attr(test_data, continuous_attr_index, get_bins=False, bins=bins_list, n=N_CLASSES)
    test_data = threshold_data(test_data, thresholds, continuous_attr_index)
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]

    Y_test[Y_test == '<=50K.'] = '<=50K'
    Y_test[Y_test == '>50K.'] = '>50K'

    print("\n\nTEST DATA\n")
    out = format(accuracy(decision_tree.predict(X_test), Y_test), '.3f')
    print("Accuracy: " + out)
    out = format(precision(decision_tree.predict(X_test), Y_test), '.3f')
    print("Precision: " + out)
    out = format(recall(decision_tree.predict(X_test), Y_test), '.3f')
    print("Recall: " + out)


if __name__ == "__main__":
    main()
