FROM python:3.7.4

ENV PYTHONUNBUFFERED 1

RUN mkdir /GPAM
WORKDIR /GPAM

COPY requirements.txt /GPAM/
RUN pip install -r requirements.txt
COPY . /GPAM/
