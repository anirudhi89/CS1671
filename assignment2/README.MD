# Assignment 2
Name: Anirudh Iyer (AKI22)
Undergraduate Section (CS1671)

## To Run:
* First, ensure that you have python 3 downloaded on your machine. Run `python3 --version` to verify
* On your terminal, run `python3 ngram_skeleton.py`. This will run the test suite.
Note: `python3` is required instead of `python` because Mac OS X comes with python 2.7.10 pre-installed. I don't believe this is required in either Windows or Linux

## Computing Environment
- Python Version: Python 3.9.6
- Operating System: Mac OS Ventura 13.2.1
- Architecture: ARM64 (M1)
This code was also tested in an Ubunutu environment (See the Dockerfile section for more information)

## Input Files
- `train/shakespeare_input.txt` for Training
- `test_data/nytimes_article.txt` for Testing
- `test_data/shakespeare_sonnets.txt` for Testing

## Dependencies
- random
    - for `random.random()`, `random.seed()`, and `random.choice()`
- math
    - for `math.log()`, and `math.exp()`

## Issues
- Some unexpected behavior discussed in Writeup.PDF
- Issues pertaining to `NgramModelWithInterpolation()`:
    - For `NgramModelWithInterpolation()`, I wasn't able to get the exact correct probability for some of the test cases shown in the document.

## Assumptions:
- `len(text)` is always greater than `c` in ngrams(`c`, `text`)
    - Thus, `len(text)` is always greater than 0 in ngrams(`c`, `text`), since we know that `c` is always greater than 0
- No preproccessing is required for the input text, including `text.lower()`
- In `random_char()`, if there is a novel context, then a random character from the vocab is used (per class discussion 10.04.23)

## Testing:
From the assignment information on Canvas, I wrote the test cases, each wrapped within the `if __name__ == "__main__":` block, so they will only execute if the file `ngram_skeleton.py` is called directly. Additioanlly, there are print statements within a few methods. To run these test cases, uncomment out the `if __name__ == "__main__":` block, and run `python3 ngram_skeletons.py` in the terminal.


## References:
- Slides from class
- Jurafsky and Martin textbook [3rd Edition]
- (Lecture 17 of Stanford NLP Course)[https://www.youtube.com/watch?v=oZHRFj8heWM&list=PLoROMvodv4rOFZnDyrlW3-nI7tMLtmiJZ&index=17&ab_channel=StanfordOnline] for help with understanding interpolation


## Docker Specifications;
- OS: Ubuntu 22.04.2 LTS
- Architecture: Multi-architecture (amd64, arm64)
- Python Version: Python 3.9.6