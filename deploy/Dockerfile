FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# tzdata so Asia/Kolkata + NSE session logic behave; curl for the healthcheck
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata curl \
 && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Kolkata

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY precision_tap ./precision_tap
COPY config.example.yaml universe README.md ANALYSIS.md INDICATOR.txt ./
RUN cp -n config.example.yaml config.yaml || true \
 && mkdir -p data/cache data/csv results logs

# the loop handles its own scheduling; no supervisor needed
CMD ["python", "-m", "precision_tap", "run"]
