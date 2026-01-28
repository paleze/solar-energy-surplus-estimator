# remove docker compose airflow setup
docker compose -f docker-compose-airflow.yaml down --volumes --remove-orphans


# Clean airflow folders
airflow_proj_dir=$(grep AIRFLOW_PROJ_DIR .env | cut -d '=' -f2)
if [ -z ${airflow_proj_dir} ]; then
    airflow_proj_dir="."
fi
airflow_dirs="dags,logs,plugins,config"
for dir in $(echo ${airflow_dirs} | tr "," "\n"); do
    if [ -d ${airflow_proj_dir}/$dir ]; then
        echo "Removing Airflow folder: ${airflow_proj_dir}/$dir"
        rm -rf ${airflow_proj_dir}/$dir
    fi
done







