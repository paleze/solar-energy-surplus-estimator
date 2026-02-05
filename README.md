# solar-energy-surplus-estimator
A ML project that predicts how much energy surplus will be produced today from my solar panels


# Initial setup
Rename the `.env.demo` file into `.env` and change environment variables if needed.

To start the airflow stack for the first time:
- optionally configure env variable `AIRFLOW_PROJ_DIR` (it defaults to the current folder if not set)
- run `setup-airflow.sh` to perform these actions:
    - set env var `AIRFLOW_UID` to the current user id and append it to a local `.env` file
    - create airflow folders according in `AIRFLOW_PROJ_DIR`

- run `docker compose up -d` to start stack containers (you can have a nice view of running containers by executing `watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"` in another terminal)
- open the `Minio`, `MlFlow`, and `Airflow` web pages on `localhost` at the corresponding configured ports

To stop the stack run `docker compose stop`

# Clenaup
To completely remove the stack run `cleanup.sh` to perform these actions:
- take the stack down and remove volumes and orphans ( executes `docker compose down --volumes --remove-orphans`)
- remove all airflow folders from `AIRFLOW_PROJ_DIR` (or from current folder if not set)

