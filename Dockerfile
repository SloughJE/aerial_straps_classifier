FROM ubuntu:20.04

RUN mkdir /code
WORKDIR /code

ENV PYTHONBUFFERED 1

RUN apt-get -y update
RUN apt-get -y install bc curl gnupg2 jq less python3 python3-pip unzip wget
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get -y install git

COPY requirements.txt /code/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
