FROM python:3.9.3-slim

RUN pip install pipenv

WORKDIR /app

COPY ["predict.py", "Pipfile", "Pipfile.lock", "./"] 

RUN pipenv install --system --deploy

RUN mkdir models

RUN mkdir logs

COPY ["models/model_xception_2024-01-06_22-44-37.keras", "models/"]

EXPOSE 6969

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:6969", "predict:app"]