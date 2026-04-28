ARG AIRFLOW_VERSION="${AIRFLOW_VERSION:?}"
ARG AIRFLOW_IMAGE_NAME="${AIRFLOW_IMAGE_NAME:?}"

FROM ${AIRFLOW_IMAGE_NAME}


COPY airflow/airflow-requirements.txt ./

COPY airflow/solar-estimator-model-dag.py \
     common/weather_data.py \
     common/utils.py \
     common/mlflow_mgr.py \
     common/minio_mgr.py /opt/airflow/dags/

# This URLLIB3 setting is necessary not to let urllib3.future required by open-meteo shadow the actual urllib3 package ( https://github.com/jawah/urllib3.future )
RUN URLLIB3_NO_OVERRIDE=1 uv pip install --no-cache niquests --no-binary urllib3-future  "apache-airflow==${AIRFLOW_VERSION}" -r ./airflow-requirements.txt



