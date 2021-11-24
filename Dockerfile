# syntax = docker/dockerfile:1.3
FROM nvcr.io/nvidia/pytorch:21.10-py3
VOLUME /data

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential curl git cmake lsb-release wget software-properties-common gdb

WORKDIR /data
COPY scripts/getdata.sh .
RUN chmod +x getdata.sh && bash ./getdata.sh /data && chmod -R 777 /data

RUN conda install -y matplotlib numpy pandas scikit-learn setuptools tqdm
RUN pip install wandb loguru transformers tqdm

# make account and homedir for specified UID and GID
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g $GID g-$GID && useradd -u $UID -g $GID -m -d /home/$UID -s /bin/bash u-$UID
RUN --mount=type=secret,id=netrc (cat /run/secrets/netrc > /home/$UID/.netrc); chown $UID:$GID /home/$UID/.netrc
USER $UID:$GID

WORKDIR /app
VOLUME /app
# COPY . .
# RUN pip install -e .
# WORKDIR /app/flexgpt/transformer-xl
# RUN python train_random_model_params.py
# WORKDIR /app