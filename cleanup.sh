set -e



# remove docker compose airflow setup
echo "Bringing down docker compose setup removing volumes and orphans..."
# get option to remove all downloaded images
if [ "$1" = "all" ]; then
    remove_all_images="all"
    echo "removing all images downloaded by docker-compose, this will require re-downloading them on next start"
else
    remove_all_images="local"
    echo "removing only images built locally, downloaded images will be kept and reused on next start"
fi
docker compose down --volumes --remove-orphans --rmi ${remove_all_images}


# Clean airflow folders
echo
echo "Removing airflow folders..."
airflow_proj_dir=$(grep AIRFLOW_PROJ_DIR .env | cut -d '=' -f2)
if [ -z ${airflow_proj_dir} ]; then
    airflow_proj_dir="."
fi

airflow_dirs="dags,logs,plugins,config"
for dir in $(echo ${airflow_dirs} | tr "," "\n"); do
    if [ -d ${airflow_proj_dir}/$dir ]; then
        echo
        echo "Removing Airflow folder: ${airflow_proj_dir}/$dir"
        rm -rf ${airflow_proj_dir}/$dir
    fi
done


echo
echo "Removing Postgres folders ..."
rm -rf ${POSTGRES_AIRFLOW_DATA_PATH:-./postgres/airflow-data}
rm -rf ${POSTGRES_MLFLOW_DATA_PATH:-./postgres/mlflow-data}


echo
echo "Removing monitoring folders ..."
rm -rf ${PROMETHEUS_DATA_PATH:-./prometheus/data}
rm -rf ${GRAFANA_DATA_PATH:-./grafana/data}


echo
echo "Removing Minio folders ..."
rm -rf ${MINIO_DATA_PATH:-./minio/data}


# remove airflow extended image
echo
echo "Removing custom images..."
airflow_extended_image=$(grep AIRFLOW_EXTENDED_IMAGE .env | cut -d '=' -f2)
if [ -z ${airflow_extended_image} ]; then
    airflow_extended_image="estimator-custom-airflow:latest"
fi
if [ ! -z $(docker images -q ${airflow_extended_image}) ]; then
    echo
    echo "Removing Airflow extended image: ${airflow_extended_image}"
    docker rmi ${airflow_extended_image}
fi


# remove app-serving custom image
app_extended_image=$(grep APP_EXTENDED_IMAGE .env | cut -d '=' -f2)
if [ -z ${app_extended_image} ]; then
    app_extended_image="estimator-app-serving:latest"
fi
if [ ! -z $(docker images -q ${app_extended_image}) ]; then
    echo
    echo "Removing model serving app image: ${app_extended_image}"
    docker rmi ${app_extended_image}
fi


echo
echo "Cleanup completed"
echo
