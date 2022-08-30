ARG BASE_CONTAINER=python:3.10-alpine
FROM $BASE_CONTAINER
LABEL author="Dominic Steinhoefel"

USER root
RUN echo "https://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories
RUN apk update
RUN apk upgrade
RUN apk add python3-dev git bash fish z3 gcc g++ libgcc make clang racket graphviz

RUN wget https://github.com/Clever/csvlint/releases/download/v0.3.0/csvlint-v0.3.0-linux-amd64.tar.gz -O /tmp/csvlint.tar.gz
RUN tar xzf /tmp/csvlint.tar.gz -C /tmp
RUN mv /tmp/csvlint-v0.3.0-linux-amd64/csvlint /usr/bin

RUN adduser -D islearn
USER islearn
WORKDIR /home/islearn
RUN pip install --upgrade pip wheel

RUN git clone https://github.com/rindPHI/islearn.git
WORKDIR /home/islearn/islearn
RUN git pull
RUN git checkout v0.2.13
RUN pip install -e .[dev,test]
RUN pip install isla-solver==0.10.6

WORKDIR /home/islearn
CMD ["fish"]
