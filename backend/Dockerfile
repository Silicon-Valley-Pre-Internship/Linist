# using base image
# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Install linux packages
RUN apt update && apt install -y screen

# Install python dependencies
RUN pip install --upgrade pip
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get update && apt-get -y install libglib2.0-0

# copy requirements.txt to current folder
COPY requirements.txt .

# installing flask requirements
RUN pip install -r requirements.txt

RUN pip install gsutil

# Create working directory
# setting working dir inside container
RUN mkdir -p /usr/src/app

# Copy contents
COPY . /usr/src/app
WORKDIR /usr/src/app/app

ENV GOOGLE_APPLICATION_CREDENTIALS="/usr/src/app/app/linist-4cc646b68cad.json"

EXPOSE 333

# serving flask backend through uWSGI server

CMD [ "python", "app.py" ]