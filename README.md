# CS1671
Repository for Projects, Assignments, and Labs for CS1671: Human Language Technologies (NLP) at the University of Pittsburgh

Each repository will contain the sourcecode, a README, input data files (if applicable), and a Dockerfile.

There is a general template for the Dockerfile in the root of this repository, but each assignment may have its own Dockerfile.

## Docker Image;
- OS: Ubuntu 22.04.2 LTS
- Architecture: Multi-architecture (amd64, arm64)
- Python Version: Python 3.11.5

To use this, first ensure docker buildx is installed. Run `docker buildx install` and `docker buildx create --use`.
Navigate to the directory with the Dockerfile, and run :`docker buildx build --platform linux/amd64,linux/arm64 -t TAGNAME-HERE .`
Then run `docker run --rm -it TAGNAME-HERE`

Keep in mind that building this docker image will take some time (25-30 minutes), because compiling Python from source is a slow process.

## Assignment 1: Regular Expression
- Use regular experessions to extract information from the dataset

## Assignment 2: N-grams
- Predict the next character in a sequence, given the previous characters.
- Optional Assignment for Masters Students: Decode Caesar Cipher using N-grams

## Assignment 3: Naive Bayes Text Classification
