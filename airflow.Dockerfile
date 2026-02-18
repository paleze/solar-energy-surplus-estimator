ARG AIRFLOW_VERSION="${AIRFLOW_VERSION:?}"
ARG AIRFLOW_IMAGE_NAME="${AIRFLOW_IMAGE_NAME:?}"

FROM ${AIRFLOW_IMAGE_NAME}:${AIRFLOW_VERSION}

COPY airflow-requirements.txt .

# This URLLIB3 setting is necessary not to let urllib3.future required by open-meteo shadow the actual urllib3 package ( https://github.com/jawah/urllib3.future )
RUN URLLIB3_NO_OVERRIDE=1 uv pip install --no-cache niquests --no-binary urllib3-future  "apache-airflow==${AIRFLOW_VERSION}" -r ./airflow-requirements.txt


