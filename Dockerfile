ARG BASE_CONTAINER=python:3.10-alpine
FROM $BASE_CONTAINER
LABEL author="Dominic Steinhoefel"

USER root
RUN echo "https://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories
RUN apk update
RUN apk upgrade
# RUN apk add python3-dev git bash fish z3 swi-prolog py3-scipy gcc g++ libgcc libquadmath gfortran libgfortran musl musl-dev make cmake libffi-dev lapack lapack-dev pkgconf zlib zlib-dev jpeg jpeg-dev zeromq zeromq-dev bash yaml yaml-dev clang racket graphviz
RUN apk add python3-dev git bash fish z3 gcc g++ libgcc make clang racket graphviz

RUN wget https://github.com/Clever/csvlint/releases/download/v0.3.0/csvlint-v0.3.0-linux-amd64.tar.gz -O /tmp/csvlint.tar.gz
RUN tar xzf /tmp/csvlint.tar.gz -C /tmp
RUN mv /tmp/csvlint-v0.3.0-linux-amd64/csvlint /usr/bin

RUN adduser -D islearn
USER islearn
WORKDIR /home/islearn
ADD requirements.txt /home/islearn

RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install -r requirements.txt

CMD ["fish"]
