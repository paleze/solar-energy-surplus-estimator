ARG PYTHON_IMAGE=${PYTHON_IMAGE:?}

# Builder image
FROM ${PYTHON_IMAGE} AS builder


# Install dependencies in the virtual environment
COPY app-serving-requirements.txt .
# This URLLIB3 setting is necessary not to let urllib3.future required by open-meteo shadow the actual urllib3 package ( https://github.com/jawah/urllib3.future )
RUN pip install uv && \
    (URLLIB3_NO_OVERRIDE=1 uv pip install --system --no-cache niquests --no-binary urllib3-future -r ./app-serving-requirements.txt) && \
    # Remove pip, setuptools, and wheel from the venv
    pip uninstall -y uv pip setuptools wheel


# Create a custom user
RUN groupadd streamlit && \
    useradd -g streamlit streamlit

# Swith to custom user and set its home directory as WORKDIR
USER streamlit
WORKDIR /home/streamlit

# copy application files
COPY app-serving.py weather_data.py metrics.py .

EXPOSE 8501