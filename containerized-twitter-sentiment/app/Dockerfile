FROM python:3.9.16-slim-buster
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3","app.py"]