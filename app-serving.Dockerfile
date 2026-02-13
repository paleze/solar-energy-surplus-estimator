ARG PYTHON_IMAGE

FROM ${PYTHON_IMAGE}


# Create a custom user
RUN groupadd streamlit && \
    useradd -g streamlit streamlit

# Swith to custom user and set its home directory as WORKDIR
USER streamlit
WORKDIR /home/streamlit

# copy application files
COPY app-serving-requirements.txt app-serving.py weather_data.py .

# This URLLIB3 setting is necessary not to let urllib3.future required by open-meteo shadow the actual urllib3 package ( https://github.com/jawah/urllib3.future )
USER root
RUN pip install uv && (URLLIB3_NO_OVERRIDE=1 uv pip install --system --no-cache niquests --no-binary urllib3-future -r ./app-serving-requirements.txt)

USER streamlit

EXPOSE 8501