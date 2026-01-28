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
airflow_exetended_image=$(grep AIRFLOW_EXTENDED_IMAGE .env | cut -d '=' -f2)
if [ -z ${airflow_exetended_image} ]; then
    airflow_exetended_image="estimator-airflow-extended:latest"
fi
if [ ! -z $(docker images -q ${airflow_exetended_image}) ]; then
    echo
    echo "Removing Airflow extended image: ${airflow_exetended_image}"
    docker rmi ${airflow_exetended_image}
fi



