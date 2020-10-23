FROM pytorch/pytorch:latest

RUN apt update && apt install -y less nano jq

COPY bash.bashrc /etc/bash.bashrc

ENV HOME=/tmp
ARG DOCKER_WORKSPACE_PATH
RUN mkdir -p $DOCKER_WORKSPACE_PATH/src
WORKDIR $DOCKER_WORKSPACE_PATH/src

RUN pip install scipy