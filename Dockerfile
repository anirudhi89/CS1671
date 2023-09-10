FROM ubuntu:latest

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary packages
RUN apt-get update && apt-get install -y \
    locales \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
&& apt-get clean && rm -rf /var/lib/apt/lists/* \
&& echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
&& locale-gen

# Set the desired Python version
ENV PYTHON_VERSION=3.11.5

# Download and install Python 3.11.5
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz \
    && tar -xf Python-${PYTHON_VERSION}.tar.xz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tar.xz

# Set the working directory inside the container
WORKDIR /app

# Copy your Python code into the container
COPY . /app

#TODO: .dockgerignore file

# Install Python dependencies - Commented out for this Assignment
# RUN pip3 install -r requirements.txt

# Command to run your Python script
CMD ["python3", "python_questions.py"]

## To build: docker buildx build --platform linux/amd64,linux/arm64 -t cs1671A1-aki22 .
## To run: docker run -rm --it cs1671A1-aki22
