# Build from the repository root (the environment reuses env.py):
#   docker build -f relief_openenv/server/Dockerfile -t reliefroute-openenv .
#   docker run -p 8000:8000 reliefroute-openenv
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "openenv>=0.9" "uvicorn[standard]>=0.30"
COPY env.py ./
COPY relief_openenv ./relief_openenv
EXPOSE 8000
CMD ["uvicorn", "relief_openenv.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
