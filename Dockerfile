ARG AIRFLOW_VERSION=3.1.6
ARG AIRFLOW_IMAGE_NAME="apache/airflow:${AIRFLOW_VERSION}"
FROM ${AIRFLOW_IMAGE_NAME}

COPY requirements.txt /

RUN uv pip install --no-cache "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt



