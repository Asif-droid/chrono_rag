FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# COPY chrono_rag.py .

# JSON input files are mounted at /data (see docker-compose.yml)
ENV DATA_DIR=/data

# CMD ["python", "chrono_rag.py"]
CMD ["bash"]