FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" websockets aio-pika numpy
COPY backend /app/backend
RUN mkdir -p /app/data
EXPOSE 8000
CMD ["python", "backend/server.py"]
