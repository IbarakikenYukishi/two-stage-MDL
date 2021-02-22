FROM selenium/standalone-chrome
USER root

RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# install requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r ./requirements.txt

# ADD . /itochu-estimator
WORKDIR /itochu-estimator
COPY . .
# COPY . /itochu-estimator
# CMD ["python", "./experiment/dow_jones.py"]