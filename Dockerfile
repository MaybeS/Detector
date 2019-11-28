FROM tiangolo/meinheld-gunicorn-flask:python3.7

COPY ./app /app
COPY ./app/config.example.json /app/config.json
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN apt update && apt install -y python-skimage
