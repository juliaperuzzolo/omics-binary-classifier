# Fixed base image: Python 3.11 (numba 0.60 supports it).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Exact dependency versions (numpy 2.0.2, numba 0.60.0, shap 0.48.0, ...).
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Code only. data/ and results/ are mounted as volumes at run time.
COPY pipeline ./pipeline
COPY tests ./tests
COPY pytest.ini ./
COPY params.yaml ./

ENTRYPOINT ["python", "-m", "pipeline"]
CMD ["--config", "params.yaml", "--validate-only"]
