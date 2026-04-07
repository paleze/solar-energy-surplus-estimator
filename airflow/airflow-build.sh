
AIRFLOW_VERSION_DEFAULT="3.1.6"
docker build \
    --build-arg AIRFLOW_VERSION=${AIRFLOW_VERSION:-${AIRFLOW_VERSION_DEFAULT}}\
    --build-arg AIRFLOW_IMAGE_NAME=${AIRFLOW_IMAGE_NAME:-apache/airflow:slim-${AIRFLOW_VERSION:-${AIRFLOW_VERSION_DEFAULT}}-python3.13} \
    -f airflow.Dockerfile \
    -t estimator-custom-airflow:latest \
    .
