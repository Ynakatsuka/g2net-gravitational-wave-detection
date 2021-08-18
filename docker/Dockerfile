ARG BASE_TAG=staging

FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04 AS nvidia
FROM gcr.io/kaggle-images/python:${BASE_TAG}

# Cuda support
COPY --from=nvidia /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/trusted.gpg /etc/apt/trusted.gpg.d/cuda.gpg
# See b/142337634#comment28
RUN sed -i 's/deb https:\/\/developer.download.nvidia.com/deb http:\/\/developer.download.nvidia.com/' /etc/apt/sources.list.d/*.list

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}

# Using cuda11.0 instead of 11.1
RUN conda remove --force -y pytorch torchvision torchaudio cpuonly
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.0 torchtext==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# requirements
ADD requirements.txt  /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
