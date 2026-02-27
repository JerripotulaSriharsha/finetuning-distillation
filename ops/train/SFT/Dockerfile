FROM python:3.10-slim

WORKDIR /app

# Copy everything from project root (one level up from ops)
COPY . .
COPY wheels/ /wheels/

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        gcc wget unzip ffmpeg libsm6 libxext6 libgl1 \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ls -R /wheels

RUN pip install --no-index --find-links=/wheels -r requirements.txt

CMD ["python3", "sft_train.py"]
