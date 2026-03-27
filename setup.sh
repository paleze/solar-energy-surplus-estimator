# *** Airflow setup ***

echo "Setup Airflow ..."

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
airflow_dirs="dags,logs,plugins,config,data"
for dir in $(echo ${airflow_dirs} | tr "," "\n"); do
    if [ ! -d ${airflow_proj_dir}/$dir ]; then
        echo "Creating missing Airflow folder: ${airflow_proj_dir}/$dir"
        mkdir -p ${airflow_proj_dir}/$dir
    fi
done

# Copy the solar estimator DAG to the Airflow dags folder
echo "Copying solar estimator DAG to Airflow dags folder..."
cp solar-estimator-model-dag.py weather_data.py ${airflow_proj_dir}/dags


# change grafana folder permissions to let grafana create folders with its grafana user
echo "Setup monitoring ..."
mkdir -p grafana/data
chmod -R o+w grafana


# Build the Airflow Docker image
echo "Building Airflow Docker image..."
./airflow-build.sh


# *** App-serving setup ***

# Build the app-serving Docker image
echo "Building app-serving image..."
./app-serving-build.sh

echo "Setup completed."
