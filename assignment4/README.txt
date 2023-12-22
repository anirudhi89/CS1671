# Assignment 4
Name: Anirudh Iyer (AKI22)
Undergraduate Section (CS1671)

## To Run:
* First, ensure that you have python 3 downloaded on your machine. Run `python3 --version` to verify
* On your terminal, run `python3 hw4_skeleton_aki22.py`. This will run the test suite.
Note: `python3` is required instead of `python` because Mac OS X comes with python 2.7.10 pre-installed. I don't believe this is required in either Windows or Linux

## Computing Environment
- Python Version: Python 3.11.6
- Operating System: Mac OS Sonoma 14.1.0
- Architecture: ARM64 (M1)
This code was also tested in an Ubunutu environment (See the Dockerfile section for more information)

## Input Files
- `vocab.txt`
- `play_names.txt`
- `will_play_text.csv`

## Dependencies
- numpy

## Issues
- Due to Python's floating point precision error's, computing cosine similarity with a word and itself often returns 0.99999999999... instead of 1.0. I suspect this doesn't have as big of an effect in the final results, but it is something to note.

## Assumptions:
- N/A

## References:
- Slides from class
- Jurafsky and Martin textbook [3rd Edition]


## Docker Specifications;
- OS: Ubuntu 22.04.2 LTS
- Architecture: Multi-architecture (amd64, arm64)
- Python Version: Python 3.9.6