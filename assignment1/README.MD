# Assignment 1
Name: Anirudh Iyer (AKI22)

## To Run:
* First, ensure that you have python 3 downloaded on your machine. Run `python3 --version` to verify
* On your terminal, run `python3 python_questions.py`. This will run the test suite (See line 104 of python_questions.py)
Note: `python3` is required instead of `python` because the code is written in python 3, and Mac OS X comes with python 2.7.10 pre-installed. I don't believe this is required in either Windows or Linux

## Computing Environment
- Python Version: Python 3.11.5
- Operating System: Mac OS X 13.2.1
- Archtecture: ARM64 (M1)
This code was also tested in an Ubunutu environment (See the Dockerfile section for more information)

## Dependencies
- re

## Issues
- None.

## Assumptions:
- For `replace_rgb()`, I am just replacing the text in the string, and not returning True/False (per the discussion in class)
- For red_reviews, and white_reviews, the a sentence (and thus its words) CAN appear in both - sets are not mutually exclusive

## Testing:
I wrote test cases for all functions, as defined in `test_suite()` in python_questions.py. These tests cases are wrapped within the `if __name__ == "__main__":` block, so they will only execute if the python_questions is called. To run the test suite, run `python3 python_questions.py` in the terminal.


## References:
- https://docs.python.org/3.4/library/re.html for regex syntax
- https://www.w3schools.com/python/python_regex.asp Metacharacters, Special Sequences and Set Syntax
- https://regex101.com/ for testing regex
- https://www.debuggex.com/ Regex visualizer
- DockerHub and StackOverflow, for the Dockerfile (for my testing - didn't want any "it runs on my machine" issues.)

## Docker Specifications;
- OS: Ubuntu 22.04.2 LTS
- Architecture: Multi-architecture (amd64, arm64)
- Python Version: Python 3.11.5