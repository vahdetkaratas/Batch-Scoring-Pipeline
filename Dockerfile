FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

COPY src ./src
COPY web ./web
COPY vercel_static ./vercel_static
COPY tests/fixtures ./tests/fixtures

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
