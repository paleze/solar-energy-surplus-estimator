
set -e

# Setup environment 
./setup.sh

# Build the Airflow Docker image
echo "Building Airflow Docker image..."
cd ${home_dir}/airflow
./airflow-build.sh
cd ${home_dir}


# Build the app-serving Docker image
echo
echo "Building app-serving image..."
cd ${home_dir}/app-serving
./app-serving-build.sh
cd ${home_dir}


# *** Run docker compose ***
echo 
echo "Starting docker stack ..."
docker compose up -d

echo
echo "Waiting for containers to become healthy ..."
time=0
timeout=60
while (( $(docker ps | grep estimator | grep -v "(healthy)" | wc -l) > 0 && $time<$timeout )); do 
    time=$((time+1))
    sleep 1
done
if [ "$time" -eq "$timeout" ];then
    echo "Error: some containers did not become healthy after $timeout seconds, check with 'docker ps'"
    echo
    exit 1
fi


# Show tab urls to open
echo
./show-tab-urls.sh

echo
echo "Setup completed."




