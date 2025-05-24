FROM python:3.10

RUN mkdir /build
COPY . /build/
WORKDIR /build/

RUN apt-get update && \
    apt-get -y install locales libgl1 && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

RUN pip install --upgrade pip
RUN pip install -r requirements.txt