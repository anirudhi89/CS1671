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

## Assignment 1: Regular Expression


## Assignment 2

