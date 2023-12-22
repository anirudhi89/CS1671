#############################################################
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    ## YOUR CODE HERE...
    true_pos = 0
    false_pos = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            true_pos += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            false_pos += 1
    if (true_pos + false_pos == 0):
        return 0
    precision = true_pos / (true_pos + false_pos)
    return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    true_pos = 0
    false_neg = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            true_pos += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            false_neg += 1
    if (true_pos + false_neg == 0):
        return 0
    recall = true_pos / (true_pos + false_neg)
    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    if (precision + recall == 0):
        return 0
    fscore = (2 * precision * recall) / (precision + recall)
    return fscore


def test_predictions(y_pred, y_true):
    '''
    Prints out the f-score, precision, and recall for the predictions.
    '''
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-score: " + str(fscore))

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

### 2.1: A very simple baseline

## Makes feature matrix for all complex
def all_complex_feature(words):
    result = []
    for word in words:
        result.append(1)
    return result

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, labels = load_file(data_file)
    y_pred = all_complex_feature(words)
    precision = get_precision(y_pred, labels)
    recall = get_recall(y_pred, labels)
    fscore = get_fscore(y_pred, labels)
    performance = [precision, recall, fscore]
    return performance


### 2.2: Word length thresholding

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    feature = []
    for word in words:
        if len(word) >= threshold:
            feature.append(1)
        else:
            feature.append(0)
    return feature

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    words, labels = load_file(training_file)
    precisions = []
    recalls = []
    fscores = []
    thresholds = range(11)
    for threshold in thresholds:
        y_pred_train = length_threshold_feature(words, threshold)
        precisions.append(get_precision(y_pred_train, labels))
        recalls.append(get_recall(y_pred_train, labels))
        fscores.append(get_fscore(y_pred_train, labels))
        # write to file, each threshold, precision, recall, fscore
        # find_threshold = "writeup_docs/2.2find_threshold.txt"
        # with open(find_threshold, "a") as threshold_name:
        #     threshold_name.write("Training Set\n")
        #     threshold_name.write("Threshold: " + str(threshold) + "\n")
        #     threshold_name.write("Precision: " + str(precisions[threshold]) + "\n")
        #     threshold_name.write("Recall: " + str(recalls[threshold]) + "\n")
        #     threshold_name.write("F-score: " + str(fscores[threshold]) + "\n")
        #     threshold_name.write("\n")
        # 
    threshold_choice = np.argmax(fscores)

    y_pred_train = length_threshold_feature(words, threshold_choice)
    tprecision = get_precision(y_pred_train, labels)
    trecall = get_recall(y_pred_train, labels)
    tfscore = get_fscore(y_pred_train, labels)
    training_performance = [tprecision, trecall, tfscore]

    words_dev, labels_dev = load_file(development_file)
    y_pred_dev = length_threshold_feature(words_dev, threshold_choice)
    dprecision = get_precision(y_pred_dev, labels_dev)
    drecall = get_recall(y_pred_dev, labels_dev)
    dfscore = get_fscore(y_pred_dev, labels_dev)

    # plot precision-recall curve
    # plt.figure(figsize=(10, 7.5))
    # plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve for Training Set')
    # plt.grid(True)
    # for i, txt in enumerate(thresholds):
    #     plt.annotate(str(txt), (recalls[i], precisions[i]))
    # plt.show()

    # write to file: file (training or test), threshold, precision, recall, fscore
    # dev_name = "writeup_docs/2.1length_threshold_dev.txt"
    # train_name = "writeup_docs/2.1length_threshold_train.txt"
    # with open(dev_name, "a") as dev_file:
    #     dev_file.write("Development Set\n")
    #     dev_file.write("Threshold: " + str(threshold_choice) + "\n")
    #     dev_file.write("Precision: " + str(dprecision) + "\n")
    #     dev_file.write("Recall: " + str(drecall) + "\n")
    #     dev_file.write("F-score: " + str(dfscore) + "\n")
    #     dev_file.write("\n")
    # with open(train_name, "a") as train_file:
    #     train_file.write("Training Set\n")
    #     train_file.write("Threshold: " + str(threshold_choice) + "\n")
    #     train_file.write("Precision: " + str(tprecision) + "\n")
    #     train_file.write("Recall: " + str(trecall) + "\n")
    #     train_file.write("F-score: " + str(tfscore) + "\n")
    #     train_file.write("\n")
    # return recalls, precisions
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
    counts = defaultdict(int) 
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t') 
            if token[0].islower(): 
                counts[token] = int(count) 
    return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    feature = []
    for word in words:
        if counts[word] < threshold:
            feature.append(1)
        else:
            feature.append(0)
    return feature

def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE

    words, labels = load_file(training_file)
    precisions = []
    recalls = []
    fscores = []
    thresholds = list(range(100000, 100000000, 100000))
    for threshold in thresholds:
        y_pred_train = frequency_threshold_feature(words, threshold, counts)
        precisions.append(get_precision(y_pred_train, labels))
        recalls.append(get_recall(y_pred_train, labels))
        fscores.append(get_fscore(y_pred_train, labels))
        # write to file, each threshold, precision, recall, fscore
        # find_threshold = "writeup_docs/2.2find_threshold.txt"
        # with open(find_threshold, "a") as threshold_name:
        #     indx = thresholds.index(threshold)
        #     threshold_name.write("Training Set\n")
        #     threshold_name.write("Threshold: " + str(threshold) + "\n")
        #     threshold_name.write("Precision: " + str(precisions[indx]) + "\n")
        #     threshold_name.write("Recall: " + str(recalls[indx]) + "\n")
        #     threshold_name.write("F-score: " + str(fscores[indx]) + "\n")
        #     threshold_name.write("\n")
    threshold_choice = thresholds[np.argmax(fscores)]
    # print(str(threshold_choice))

    y_pred_train = frequency_threshold_feature(words, threshold_choice, counts)
    tprecision = get_precision(y_pred_train, labels)
    trecall = get_recall(y_pred_train, labels)
    tfscore = get_fscore(y_pred_train, labels)
    training_performance = [tprecision, trecall, tfscore]

    words_dev, labels_dev = load_file(development_file)
    y_pred_dev = frequency_threshold_feature(words_dev, threshold_choice, counts)
    dprecision = get_precision(y_pred_dev, labels_dev)
    drecall = get_recall(y_pred_dev, labels_dev)
    dfscore = get_fscore(y_pred_dev, labels_dev)

    # plot precision-recall curve
    # plt.figure(figsize=(10, 7.5))
    # plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve for Training Set')
    # plt.grid(True)
    # # for i, txt in enumerate(thresholds):
    # #     plt.annotate(str(txt), (recalls[i], precisions[i]))
    # plt.show()

    # write to file: file (training or test), threshold, precision, recall, fscore
    # dev_name = "writeup_docs/2.2length_threshold_dev.txt"
    # train_name = "writeup_docs/2.2length_threshold_train.txt"
    # with open(dev_name, "a") as dev_file:
    #     dev_file.write("Development Set\n")
    #     dev_file.write("Threshold: " + str(threshold_choice) + "\n")
    #     dev_file.write("Precision: " + str(dprecision) + "\n")
    #     dev_file.write("Recall: " + str(drecall) + "\n")
    #     dev_file.write("F-score: " + str(dfscore) + "\n")
    #     dev_file.write("\n")
    # with open(train_name, "a") as train_file:
    #     train_file.write("Training Set\n")
    #     train_file.write("Threshold: " + str(threshold_choice) + "\n")
    #     train_file.write("Precision: " + str(tprecision) + "\n")
    #     train_file.write("Recall: " + str(trecall) + "\n")
    #     train_file.write("F-score: " + str(tfscore) + "\n")
    #     train_file.write("\n")
    # return recalls, precisions
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.4: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE    
    words, labels = load_file(training_file)
    precisions = []
    recalls = []
    fscores = []
    thresholds = range(11)
    for threshold in thresholds:
        y_pred_train = length_threshold_feature(words, threshold)
        precisions.append(get_precision(y_pred_train, labels))
        recalls.append(get_recall(y_pred_train, labels))
        fscores.append(get_fscore(y_pred_train, labels))
    threshold_choice_length = np.argmax(fscores)
    train_length_feat = length_threshold_feature(words, threshold_choice_length)
    
    precisions = []
    recalls = []
    fscores = []
    thresholds = list(range(100000, 100000000, 100000))
    for threshold in thresholds:
        y_pred_train = frequency_threshold_feature(words, threshold, counts)
        precisions.append(get_precision(y_pred_train, labels))
        recalls.append(get_recall(y_pred_train, labels))
        fscores.append(get_fscore(y_pred_train, labels))
    threshold_choice_freq = thresholds[np.argmax(fscores)]
    train_freq_feat = frequency_threshold_feature(words, threshold_choice_freq, counts)
    
    # x = m x n
    # y = m
    X_train = np.column_stack((train_length_feat, train_freq_feat))
    X_train = ((X_train - X_train.mean(axis = 0)) / X_train.std(axis = 0))
    Y = np.asarray(labels)

    clf = GaussianNB()
    clf.fit(X_train, Y)
    Y_pred_train = clf.predict(X_train)
    tprecision = get_precision(Y_pred_train.tolist(), Y.tolist())
    trecall = get_recall(Y_pred_train.tolist(), Y.tolist())
    tfscore = get_fscore(Y_pred_train.tolist(), Y.tolist())

    words_dev, labels_dev = load_file(development_file)
    dev_length_feat = length_threshold_feature(words_dev, threshold_choice_length)
    dev_freq_feat = frequency_threshold_feature(words_dev, threshold_choice_freq, counts)
    X_dev = np.column_stack((dev_length_feat, dev_freq_feat))
    X_dev = ((X_dev - X_train.mean(axis = 0)) / X_train.std(axis = 0))
    Y_dev = np.asarray(labels_dev)

    Y_pred_dev = clf.predict(X_dev)
    dprecision = get_precision(Y_pred_dev.tolist(), Y_dev.tolist())
    drecall = get_recall(Y_pred_dev.tolist(), Y_dev.tolist())
    dfscore = get_fscore(Y_pred_dev.tolist(), Y_dev.tolist())
    
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance, training_performance

### 2.5: Logistic Regression

## Trains a Logistic Regression classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE
    words, labels = load_file(training_file)
    X_train = np.zeros((len(words), 2))
    Y = np.asarray(labels)
    for i, row in enumerate(X_train):
        row[0] = len(words[i])
        row[1] = counts[words[i]]
    mean1, mean2 = np.mean(X_train, axis = 0)
    std1, std2 = np.std(X_train, axis = 0)

    X_train = (X_train - np.array([mean1, mean2])) / np.array([std1, std2])
    clf = LogisticRegression()
    clf.fit(X_train, Y)
    Y_pred_train = clf.predict(X_train)
    tprecision = get_precision(Y_pred_train.tolist(), Y.tolist())
    trecall = get_recall(Y_pred_train.tolist(), Y.tolist())
    tfscore = get_fscore(Y_pred_train.tolist(), Y.tolist())

    words_dev, labels_dev = load_file(development_file)
    X_dev = np.zeros((len(words_dev), 2))
    Y_dev = np.asarray(labels_dev)
    for i, row in enumerate(X_dev):
        row[0] = len(words_dev[i])
        row[1] = counts[words_dev[i]]
    
    X_dev = (X_dev - np.array([mean1, mean2])) / np.array([std1, std2])
    Y_pred_dev = clf.predict(X_dev)
    dprecision = get_precision(Y_pred_dev.tolist(), Y_dev.tolist())
    drecall = get_recall(Y_pred_dev.tolist(), Y_dev.tolist())
    dfscore = get_fscore(Y_pred_dev.tolist(), Y_dev.tolist())

    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance, training_performance



# =================================== HELPER METHODS FOR WRITEUP ===========================================

# PR-Code for both baseline classifiers on dataset.
def pr_plot_both(dev_file, counts):

    words, labels = load_file(dev_file)
    f_precisions = []
    f_recalls = []
    f_fscore = []
    f_thresholds = list(range(100000, 100000000, 100000))
    for threshold in f_thresholds:
        y_pred_dev = frequency_threshold_feature(words, threshold, counts)
        f_precisions.append(get_precision(y_pred_dev, labels))
        f_recalls.append(get_recall(y_pred_dev, labels))
        f_fscore.append(get_fscore(y_pred_dev, labels))
    # print("Best F-score for Frequency Threshold: " + str(max(f_fscore)))
    
    l_precisions = []
    l_recalls = []
    l_fscore = []
    l_thresholds = range(11)
    for threshold in l_thresholds:
        y_pred_dev = length_threshold_feature(words, threshold)
        l_precisions.append(get_precision(y_pred_dev, labels))
        l_recalls.append(get_recall(y_pred_dev, labels))
        l_fscore.append(get_fscore(y_pred_dev, labels))
    # print("Best F-score for Length Threshold: " + str(max(l_fscore)))
    
    plt.figure(figsize=(10, 7.5))
    plt.plot(l_recalls, l_precisions, label='Word Length Threshold', marker='o')
    plt.plot(f_recalls, f_precisions, label='Word Frequency Threshold', marker='x')
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Word Length and Word Frequency Thresholds - Development Set')
    plt.legend()
    plt.show()

# WRITEUP CODE:

def baseline_20(path, file_name):
    with open(path, encoding='utf-8', errors='ignore') as f:
        file_content = f.read() 
        title = ''
        if path == 'data/complex_words_training.txt':
            title = '2.0 Baseline Training Set'
            file_name += '_training.txt'
        else:
            title = '2.0 Baseline Development Set'
            file_name += 'dev.txt'
        list_all_complex = all_complex(path)
        output_file = open(file_name, "a")
        output_file.write(title + "\n")
        output_file.write("\n")
        output_file.write("File Name: " + str(path) + "\n")
        for i in range(len(list_all_complex)):
            output_file.write(str(list_all_complex[i])+"\n")
        output_file.write("\n")
        output_file.close()


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    word_frequency_threshold(training_file, development_file, counts)