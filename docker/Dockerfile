FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY terry ./terry

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 9021

CMD ["terry", "serve", "--port", "9021", "--project", "/app"]
