FROM python:3.9

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install poetry

COPY ./poetry.lock ./pyproject.toml /app/

COPY ./src /app
COPY config.yaml /app


RUN poetry config virtualenvs.create false \
  && poetry install $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi

CMD uvicorn app:app --host 0.0.0.0 --port 80