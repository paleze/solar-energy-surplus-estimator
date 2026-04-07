set -e

home_dir=$(pwd)

# *** Airflow setup ***
echo "Setting up Airflow ..."

# Performs the manual setup steps for Airflow project described at https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html

# Add user id to .env file
if [ ! -f .env -o -z $(grep AIRFLOW_UID .env) ]; then
    uid=$(id -u)
    echo "AIRFLOW_UID not found in .env, setting to current user id: $uid"
    echo "AIRFLOW_UID=$uid" >> .env
fi

# get AIRFLOW_PROJ_DIR from .env file, default to current directory if not set
airflow_proj_dir=$(grep AIRFLOW_PROJ_DIR .env | cut -d '=' -f2)
if [ -z ${airflow_proj_dir} ]; then
    airflow_proj_dir="."
fi


# create Airflow folders
airflow_dirs="dags,logs,plugins,config"
for dir in $(echo ${airflow_dirs} | tr "," "\n"); do
    if [ ! -d ${airflow_proj_dir}/$dir ]; then
        echo "Creating missing Airflow folder: ${airflow_proj_dir}/$dir"
        mkdir -p ${airflow_proj_dir}/$dir
    fi
done

# Create postegres data folder
mkdir -p ${POSTGRES_AIRFLOW_DATA_PATH:-./postgres/airflow-data}


# Build the Airflow Docker image
echo "Building Airflow Docker image..."
cd ${home_dir}/airflow
./airflow-build.sh


# *** MLFlow setup ***
# Create postegres data folder
echo "Setting up Minio ..."
mkdir -p ${POSTGRES_MLFLOW_DATA_PATH:-./postgres/mlflow-data}


# *** Minio Setup ***
echo "Setting up Minio ..."
mkdir ${MINIO_DATA_PATH:-./minio/data}



# *** Monitoring setup ***
echo "Setting up monitoring ..."
# create monitoring folders and change permissions
mkdir -p ${PROMETHEUS_DATA_PATH:-./prometheus/data}
chmod -R o+w ${PROMETHEUS_DATA_PATH:-./prometheus/data}
mkdir -p ${GRAFANA_DATA_PATH:-./grafana/data}
chmod -R o+w ${GRAFANA_DATA_PATH:-./grafana/data}


# *** App-serving setup ***

# Build the app-serving Docker image
echo "Building app-serving image..."
cd ${home_dir}/app-serving
./app-serving-build.sh




echo "Setup completed."


