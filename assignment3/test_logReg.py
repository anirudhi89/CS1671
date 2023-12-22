
if __name__ == '__main__':
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    development_file = "data/complex_words_development.txt"
    training_file = "data/complex_words_training.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    words, labels = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    
    words += words_dev
    labels += labels_dev
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
    
    test_words = []
    with open(test_file, 'rt', encoding = "utf8") as f:
        i = 0
        for line in f:
            if i > 0: # skips over column headers
                line_split = line[:-1].split("\t")
                test_words.append(line_split[0].lower())
            i += 1

    # Extract features for the test data
    X_test = np.array([[len(word), counts[word]] for word in test_words])

    # Normalize the test data using the same means and standard deviations as the training data
    X_test = (X_test - np.array([mean1, mean2])) / np.array([std1, std2])

    # Use the trained logistic regression classifier to predict labels for the test data
    Y_test_pred = clf.predict(X_test)

    # Save the predictions in a CSV file
    import pandas as pd

    output_filename = "predict_aki22.csv"

    df = pd.DataFrame({'WORD': test_words, 'PREDICTION': Y_test_pred})
    df.to_csv(output_filename, index=False)