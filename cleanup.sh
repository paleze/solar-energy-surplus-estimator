# remove docker compose airflow setup
echo "Bringing down docker compose setup removing volumes and orphans..."
docker compose down --volumes --remove-orphans


# Clean airflow folders
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


# remove airflow extended image
airflow_extended_image=$(grep AIRFLOW_EXTENDED_IMAGE .env | cut -d '=' -f2)
if [ -z ${airflow_extended_image} ]; then
    airflow_extended_image="estimator-airflow-extended:latest"
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

