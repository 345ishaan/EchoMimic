FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS base

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=America/New_York

# Install tzdata and other packages
RUN apt-get update && \
    apt-get install -y tzdata git git-lfs && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /EchoMimic

# RUN git lfs install
# RUN git clone https://huggingface.co/BadToBest/EchoMimic pretrained_weights

FROM base AS final

COPY . /EchoMimic

RUN pip install -r requirements.txt

RUN apt-get install -y wget && wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && tar -xvf ffmpeg-release-amd64-static.tar.xz
ENV FFMPEG_PATH=/EchoMimic/ffmpeg-7.0.1-amd64-static
