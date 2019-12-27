FROM python:3.6.2

MAINTAINER nsemble.ai@gmail.com
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
