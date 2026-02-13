ARG AIRFLOW_VERSION
ARG AIRFLOW_IMAGE_NAME
FROM ${AIRFLOW_IMAGE_NAME}

# Create a custom user
# RUN groupadd airflow && \
#     useradd -g airflow airflow

# Swith to custom user and set its home directory as WORKDIR
# USER airflow
# WORKDIR /home/airflow

COPY airflow-requirements.txt weather_data.py .

#USER root

# This URLLIB3 setting is necessary not to let urllib3.future required by open-meteo shadow the actual urllib3 package ( https://github.com/jawah/urllib3.future )
RUN URLLIB3_NO_OVERRIDE=1 uv pip install --no-cache niquests --no-binary urllib3-future  "apache-airflow==${AIRFLOW_VERSION}" -r ./airflow-requirements.txt

#USER airflow
