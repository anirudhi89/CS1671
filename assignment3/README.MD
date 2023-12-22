# Assignment 3
Name: Anirudh Iyer (AKI22)
Undergraduate Section (CS1671)

## To Run:
* First, ensure that you have python 3 downloaded on your machine. Run `python3 --version` to verify
* On your terminal, run `python3 hw3_skeleton_aki22.py`. This will run the program on the default dataset, as seen on line 450
Note: `python3` is required instead of `python` because Mac OS X comes with python 2.7.10 pre-installed. I don't believe this is required in either Windows or Linux

## Computing Environment
- Python Version: Python 3.9.6
- Operating System: Mac OS Ventura 13.2.1
- Architecture: ARM64 (M1)
This code was also tested in an Ubunutu environment (See the Dockerfile section for more information)

## Input Files
- `data/complex_words_development` for Development
- `data/complex_words_training` for Training
- `data/complex_words_unlabeled` for Testing
- `ngram_counts.txt` for N-gram counts

## Dependencies
Run `pip install -r requirements.txt` to install the following dependencies:
- numpy
    - for `numpy.asarray`, `numpy.std`, `numpy.column_stack`, `numpy.mean`, `numpy.zeros`, `np.argmax`
- scikit-learn
    - for `GaussianNB` and `LogisticRegression`
    - for verifiying the accuracy of the model
- matplotlib
    - for plotting Precision-Recall Curve

## Issues
- None

## References:
- Slides from class
- Jurafsky and Martin textbook [3rd Edition]
- (Gaussian NB Docs)[https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html]
- (Logistic Regression Docs)[https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html]


## Docker Specifications;
- OS: Ubuntu 22.04.2 LTS
- Architecture: Multi-architecture (amd64, arm64)
- Python Version: Python 3.9.6