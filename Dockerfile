FROM python:3.8-slim-buster

WORKDIR /docker

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY 

CMD uvicorn yolov7.api:app --host 0.0.0.0 --port $PORT
