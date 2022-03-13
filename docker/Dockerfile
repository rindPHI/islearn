ARG BASE_CONTAINER=python:3.10.2-alpine3.15
FROM $BASE_CONTAINER
LABEL author="Dominic Steinhoefel"

USER root
RUN echo "https://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories
RUN apk update
RUN apk upgrade
RUN apk add python3-dev git swi-prolog py3-scipy gcc g++ libgcc libquadmath gfortran libgfortran musl musl-dev make cmake libffi-dev lapack lapack-dev pkgconf zlib zlib-dev jpeg jpeg-dev zeromq zeromq-dev bash yaml yaml-dev clang racket graphviz

RUN adduser -D islearn
USER islearn
WORKDIR /home/islearn
ADD requirements.txt /home/islearn

RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install --no-deps -r requirements.txt
RUN pip install z3-solver==4.8.14

CMD ["bash"]
